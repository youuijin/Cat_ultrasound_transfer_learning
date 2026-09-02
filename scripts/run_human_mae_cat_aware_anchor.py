"""Target-aware unlabeled-Cat preservation during intermediate Human MAE."""
from __future__ import annotations

import argparse, csv, json, statistics, subprocess, sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from scripts import run_cat_cross_species_anchor_validation as catrun
from scripts import run_human_mae_feature_anchor as anchor
from src.human_ssl.cat_anchor import build_cat_anchor_loader, cat_anchor_split, write_subject_csv
from src.human_ssl.data import build_ssl_loaders, discover_ssl_samples, split_ssl_samples
from src.human_ssl.feature_anchor import ANCHOR_BLOCKS, final_feature_drift
from src.human_ssl.mae import VisionMAE

CAT_RUNS=(("cat_anchor_0p003",0.003),("cat_anchor_0p01",0.01),("cat_anchor_0p03",0.03))
ALL_BLOCKS=tuple(range(12))


def args_parse():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--mode",choices=("feasibility","reproduce-0p03"),default="feasibility")
    p.add_argument("--python",default=sys.executable); p.add_argument("--cat-data-root",type=Path,default=Path("data/cat_dataset"))
    p.add_argument("--runs-dir",type=Path,default=Path("runs/human_mae_cat_aware_anchor")); p.add_argument("--cat-runs-dir",type=Path,default=Path("runs/cat_cat_aware_anchor_validation"))
    p.add_argument("--results-dir",type=Path,default=Path("results/cat_aware_anchor_feasibility")); p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--force",action="store_true"); p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True)
    return p.parse_args()


def run(cmd,label): print(f"\n{'='*72}\n{label}\n{'='*72}\n{subprocess.list2cmdline(cmd)}",flush=True); subprocess.run(cmd,cwd=ROOT,check=True)
def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def rows(p): return list(csv.DictReader(Path(p).open(encoding="utf-8")))
def write_csv(p,data):
    with Path(p).open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def expected(source,lambda_cat,seed=0,cat_data_root=Path("data/cat_dataset")):
    class A: epochs=50
    value=anchor.mae_expected(source,0.01,A(),seed,ALL_BLOCKS)
    value.update(cat_anchor_lambda=lambda_cat,cat_data_root=str(cat_data_root),cat_num_folds=5,cat_fold=0,cat_split_seed=42)
    return value


def ssl_command(a,source,name,lambda_cat,root,seed=0):
    class B: pass
    b=B(); b.python=a.python; b.epochs=50; b.num_workers=a.num_workers; b.amp=a.amp
    cmd=anchor.mae_command(b,source,name,0.01,root,seed,ALL_BLOCKS)
    cmd.extend(["--cat-anchor-lambda",str(lambda_cat),"--cat-data-root",str(a.cat_data_root),
                "--cat-num-folds","5","--cat-fold","0","--cat-split-seed","42"])
    return cmd


def probe_command(a,source,checkpoint,root,config,seed=0):
    class B: pass
    b=B(); b.python=a.python; b.num_workers=a.num_workers; b.amp=a.amp
    return anchor.probe_command(b,source,checkpoint,root,config,seed)


def best_human(root):
    valid=[r for r in rows(root/"metrics.csv") if r["phase"]=="validation"]
    r=max(valid,key=lambda x:float(x["mean_dice"])); return float(r["mean_dice"]),float(r["kidney_iou"]),float(r["loss"])


def best_cat(root):
    r=max(rows(root/"metrics.csv"),key=lambda x:float(x["validation_mean_foreground_dice"]))
    return float(r["validation_mean_foreground_dice"]),float(r["validation_mean_foreground_iou"]),float(r["validation_loss"])


def mean_drift(path): return statistics.mean(float(r["drift_1_minus_cka"]) for r in rows(path) if int(r["layer"].split("_")[-1]) in ANCHOR_BLOCKS)


def diagnostic(a,source,checkpoint,human_path,cat_path,seed=0):
    if human_path.is_file() and cat_path.is_file() and not a.force: return
    model=VisionMAE("vit_b16",source["decoder_dim"],source["decoder_depth"],source["decoder_heads"],source["norm_pixel_loss"])
    teacher=deepcopy(model.encoder).requires_grad_(False).eval(); payload=torch.load(checkpoint,map_location="cpu",weights_only=False); model.encoder.load_state_dict(payload["state_dict"],strict=True)
    samples=discover_ssl_samples({n:Path(source[f"{n}_root"]) for n in ("human1","human2","human3")}); _train,val=split_ssl_samples(samples,source["val_fraction"],seed)
    _,human_loader=build_ssl_loaders(val,val,model.image_size,source["batch_size"],a.num_workers,seed)
    cat_loader,_ct,_cv=build_cat_anchor_loader(a.cat_data_root,model.image_size,source["batch_size"],a.num_workers,seed,5,0,42,False)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); teacher.to(device)
    final_feature_drift(model,teacher,human_loader,device,human_path,ANCHOR_BLOCKS); final_feature_drift(model,teacher,cat_loader,device,cat_path,ANCHOR_BLOCKS)


def cat_location(base,method,seed=0):
    return base/method/"segmentation"/"vit_b16"/"full"/"fold_0"/f"seed_{seed}"/("init_human_mae" if method!="imagenet" else "")


def cat_command(a,method,checkpoint,seed=0):
    cmd=[a.python,"-m","src.segmentation.train","--encoder","vit_b16","--encoder-init","imagenet" if method=="imagenet" else "human_mae","--transfer","full","--data-root",str(a.cat_data_root),"--num-folds","5","--fold","0","--split-seed","42","--seed",str(seed),"--batch-size","8","--epochs","50","--lr","1e-4","--weight-decay","1e-4","--num-workers",str(a.num_workers),"--output-dir",str(a.cat_runs_dir/method),"--amp" if a.amp else "--no-amp"]
    if checkpoint: cmd.extend(["--encoder-checkpoint",str(Path(checkpoint).resolve())])
    return cmd


def reproduce_0p03(a,source,train_ids,val_ids):
    output=Path("results/cat_aware_anchor_0p03_reproducibility"); output.mkdir(parents=True,exist_ok=True)
    diagnostics=a.runs_dir/"reproduce_0p03_diagnostics"; diagnostics.mkdir(parents=True,exist_ok=True)
    write_subject_csv(output/"cat_anchor_train_subjects.csv",cat_anchor_split(a.cat_data_root,5,0,42)[0])
    write_subject_csv(output/"cat_heldout_val_subjects.csv",cat_anchor_split(a.cat_data_root,5,0,42)[1])
    methods=("imagenet","full_human_mae","pretrained_anchor","cat_aware_anchor_0p03")
    records=[]; new_ssl=[]; reused_ssl=[]; new_cat=[]; reused_cat=[]; checkpoints={}
    for seed in (0,1,2):
        full=catrun.validate_human_ssl(catrun.full_ssl(seed),seed,False)
        pretrained=catrun.validate_human_ssl(catrun.anchor_ssl(seed),seed,True)
        checkpoints[("imagenet",seed)]=None; checkpoints[("full_human_mae",seed)]=full; checkpoints[("pretrained_anchor",seed)]=pretrained
        cat_aware=(a.runs_dir/"lambda_cat_0p03"/"seed0" if seed==0 else a.runs_dir/"lambda_cat_0p03"/f"seed{seed}")
        complete=(anchor.is_run_complete(cat_aware,expected(source,.03,seed,a.cat_data_root),50) and
                  (cat_aware/"cat_train_feature_drift.csv").is_file())
        if a.force or not complete:
            run(ssl_command(a,source,f"cat_aware_anchor_0p03_seed{seed}",.03,cat_aware,seed),f"Cat-aware Human MAE 0.03 seed {seed}")
            if not (anchor.is_run_complete(cat_aware,expected(source,.03,seed,a.cat_data_root),50) and (cat_aware/"cat_train_feature_drift.csv").is_file()): raise RuntimeError(f"Incomplete Cat-aware SSL seed {seed}")
            new_ssl.append(f"seed{seed}")
        else: reused_ssl.append(f"seed{seed}")
        checkpoints[("cat_aware_anchor_0p03",seed)]=(cat_aware/"last_encoder.pt").resolve()
        probe=cat_aware/"human_frozen_probe"
        if a.force or not anchor.probe_complete(probe,anchor.probe_expected("human_mae",seed)):
            run(probe_command(a,source,cat_aware/"last_encoder.pt",probe,cat_aware/"config.json",seed),f"Human frozen probe Cat-aware 0.03 seed {seed}")
        if not anchor.probe_complete(probe,anchor.probe_expected("human_mae",seed)): raise RuntimeError(f"Incomplete Human probe seed {seed}")

    cat_roots={}
    for seed in (0,1,2):
        for method in methods:
            checkpoint=checkpoints[(method,seed)]
            if method in ("imagenet","full_human_mae","pretrained_anchor"):
                old={"imagenet":"imagenet","full_human_mae":"human_mae_full","pretrained_anchor":"human_mae_anchor_all"}[method]
                root=catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),old,seed)
                expected_cat=catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),old,seed,None if checkpoint is None else Path(checkpoint).resolve())
            else:
                root=cat_location(a.cat_runs_dir,"cat_anchor_0p03",seed)
                expected_cat=catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),method,seed,Path(checkpoint).resolve())
            if not a.force and catrun.cat_complete(root,expected_cat): reused_cat.append(f"{method}/seed{seed}")
            else:
                output_method="cat_anchor_0p03" if method=="cat_aware_anchor_0p03" else method
                root=cat_location(a.cat_runs_dir,output_method,seed)
                run(cat_command(a,output_method,checkpoint,seed),f"Cat segmentation {method} seed {seed}")
                if not catrun.cat_complete(root,expected_cat): raise RuntimeError(f"Incomplete Cat downstream {method} seed {seed}")
                new_cat.append(f"{method}/seed{seed}")
            cat_roots[(method,seed)]=root

    for seed in (0,1,2):
        for method in methods:
            checkpoint=checkpoints[(method,seed)]; human_drift=cat_drift=0.0 if method=="imagenet" else None
            if method=="imagenet": human_dice=best_human(anchor.repro_paths(seed)["imagenet"])[0]
            else:
                if method=="full_human_mae":
                    ssl_root=catrun.full_ssl(seed)
                    probe=anchor.repro_paths(seed)["full_probe"]
                elif method=="pretrained_anchor": ssl_root=catrun.anchor_ssl(seed); probe=ssl_root/"human_frozen_probe"
                else:
                    ssl_root=(a.runs_dir/"lambda_cat_0p03"/"seed0" if seed==0 else a.runs_dir/"lambda_cat_0p03"/f"seed{seed}"); probe=ssl_root/"human_frozen_probe"
                human_path=(ssl_root/"final_feature_drift.csv" if (ssl_root/"final_feature_drift.csv").is_file() else diagnostics/f"{method}_seed{seed}_human.csv")
                cat_path=(ssl_root/"cat_train_feature_drift.csv" if (ssl_root/"cat_train_feature_drift.csv").is_file() else diagnostics/f"{method}_seed{seed}_cat.csv")
                diagnostic(a,source,checkpoint,human_path,cat_path,seed); human_drift=mean_drift(human_path); cat_drift=mean_drift(cat_path); human_dice=best_human(probe)[0]
            cdice,ciou,closs=best_cat(cat_roots[(method,seed)])
            records.append({"method":method,"seed":seed,"lambda_pretrained":.01 if method in ("pretrained_anchor","cat_aware_anchor_0p03") else "","lambda_cat":.03 if method=="cat_aware_anchor_0p03" else 0 if method=="pretrained_anchor" else "","human_ssl_checkpoint":"imagenet_pretrained" if checkpoint is None else str(checkpoint),"cat_best_checkpoint":str((cat_roots[(method,seed)]/"best.pt").resolve()),"human_frozen_dice":human_dice,"mean_human_feature_drift":human_drift,"mean_cat_train_feature_drift":cat_drift,"cat_val_dice":cdice,"cat_val_iou":ciou,"cat_val_loss":closs,"reused_human_ssl":method!="cat_aware_anchor_0p03" or f"seed{seed}" in reused_ssl,"reused_cat_run":f"{method}/seed{seed}" in reused_cat,"human_run_complete":True,"cat_run_complete":True})
    write_csv(output/"all_seed_results.csv",records); index={(r["method"],r["seed"]):r for r in records}; summaries=[]
    for method in methods:
        selected=[index[(method,s)] for s in (0,1,2)]; item={"method":method,"n_seeds":3}
        for field,label in (("cat_val_dice","cat_dice"),("cat_val_iou","cat_iou"),("cat_val_loss","cat_loss"),("human_frozen_dice","human_frozen_dice"),("mean_human_feature_drift","human_feature_drift"),("mean_cat_train_feature_drift","cat_train_feature_drift")):
            mean,std=anchor.mean_std(r[field] for r in selected); item[f"mean_{label}"]=mean; item[f"std_{label}"]=std
        summaries.append(item)
    write_csv(output/"method_summary.csv",summaries); comparisons={}
    for base,stem in (("imagenet","cat_aware_vs_imagenet"),("pretrained_anchor","cat_aware_vs_pretrained_anchor"),("full_human_mae","cat_aware_vs_full")):
        paired=[]
        for seed in (0,1,2):
            b=index[(base,seed)]; c=index[("cat_aware_anchor_0p03",seed)]; prefix={"imagenet":"imagenet","pretrained_anchor":"pretrained_anchor","full_human_mae":"full"}[base]
            row={"seed":seed,f"{prefix}_dice":b["cat_val_dice"],"cat_aware_dice":c["cat_val_dice"],"delta_dice":c["cat_val_dice"]-b["cat_val_dice"]}
            if base=="imagenet": row.update(imagenet_iou=b["cat_val_iou"],cat_aware_iou=c["cat_val_iou"],delta_iou=c["cat_val_iou"]-b["cat_val_iou"])
            if base=="pretrained_anchor": row.update(pretrained_anchor_cat_drift=b["mean_cat_train_feature_drift"],cat_aware_cat_drift=c["mean_cat_train_feature_drift"],drift_reduction=b["mean_cat_train_feature_drift"]-c["mean_cat_train_feature_drift"])
            paired.append(row)
        write_csv(output/f"{stem}.csv",paired); summary=catrun.delta_summary(paired,base=="imagenet"); write_csv(output/f"{stem}_summary.csv",[summary]); comparisons[base]=paired
    labels=("ImageNet","Full Human MAE","Pretrained anchor","Cat-aware anchor")
    fig,ax=plt.subplots()
    for seed in (0,1,2): ax.plot(labels,[index[(m,seed)]["cat_val_dice"] for m in methods],marker="o",label=f"seed {seed}")
    ax.set(ylabel="Cat validation Dice"); ax.legend(); fig.tight_layout(); fig.savefig(output/"cat_aware_0p03_seed_reproducibility.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots()
    for base,p in comparisons.items(): ax.plot([r["seed"] for r in p],[r["delta_dice"] for r in p],marker="o",label=f"Cat-aware - {base}")
    ax.axhline(0,color="black",linestyle="--"); ax.set(xlabel="Seed",ylabel="Paired Dice delta",xticks=[0,1,2]); ax.legend(); fig.tight_layout(); fig.savefig(output/"cat_aware_0p03_paired_delta.png",dpi=200); plt.close(fig)
    aware=[index[("cat_aware_anchor_0p03",s)] for s in (0,1,2)]; fig,ax=plt.subplots(); ax.plot([0,1,2],[r["mean_cat_train_feature_drift"] for r in aware],marker="o",label="Cat train"); ax.plot([0,1,2],[r["mean_human_feature_drift"] for r in aware],marker="o",label="Human"); ax.set(xlabel="Seed",ylabel="Mean feature drift",xticks=[0,1,2]); ax.legend(); fig.tight_layout(); fig.savefig(output/"cat_aware_drift_reproducibility.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots()
    for method in ("pretrained_anchor","cat_aware_anchor_0p03"):
        selected=[index[(method,s)] for s in (0,1,2)]; ax.scatter([r["mean_cat_train_feature_drift"] for r in selected],[r["cat_val_dice"] for r in selected],label=method)
    ax.set(xlabel="Mean Cat-train feature drift",ylabel="Cat validation Dice"); ax.legend(); fig.tight_layout(); fig.savefig(output/"cat_drift_vs_cat_dice_0p03_reproducibility.png",dpi=200); plt.close(fig)
    print("Method                     Mean Cat Dice     SD\n------------------------------------------------")
    for r in summaries: print(f"{r['method']:<26} {r['mean_cat_dice']:.6f}  {r['std_cat_dice']:.6f}")
    for base,p in comparisons.items(): print(f"Cat-aware - {base}: "+", ".join(f"seed{r['seed']}={r['delta_dice']:.6f}" for r in p)+f", mean={statistics.mean(r['delta_dice'] for r in p):.6f}")
    print("Mean Cat-train drift: pretrained anchor "+f"{statistics.mean(index[('pretrained_anchor',s)]['mean_cat_train_feature_drift'] for s in (0,1,2)):.6f}, cat-aware 0.03 {statistics.mean(index[('cat_aware_anchor_0p03',s)]['mean_cat_train_feature_drift'] for s in (0,1,2)):.6f}")
    print("New Human SSL: "+", ".join(new_ssl)); print("Reused Human SSL: "+", ".join(reused_ssl)); print("New Cat runs: "+", ".join(new_cat)); print("Reused Cat runs: "+", ".join(reused_cat))


def main():
    a=args_parse(); a.runs_dir.mkdir(parents=True,exist_ok=True); a.cat_runs_dir.mkdir(parents=True,exist_ok=True); a.results_dir.mkdir(parents=True,exist_ok=True)
    source=read_json("checkpoints/human_mae_vit_b16_trajectory/config.json"); train_subjects,val_subjects=cat_anchor_split(a.cat_data_root,5,0,42)
    train_ids={x.subject_id for x in train_subjects}; val_ids={x.subject_id for x in val_subjects}; assert not train_ids&val_ids
    write_subject_csv(a.results_dir/"cat_anchor_train_subjects.csv",train_subjects); write_subject_csv(a.results_dir/"cat_heldout_val_subjects.csv",val_subjects)
    print(f"Human subjects/images: existing seed0 split; Cat anchor train subjects: {len(train_ids)}; Cat downstream validation subjects: {len(val_ids)}; intersection(train Cat anchor, Cat val) = {len(train_ids&val_ids)}")
    if a.mode=="reproduce-0p03":
        reproduce_0p03(a,source,train_ids,val_ids)
        return
    ssl={"pretrained_anchor":Path("runs/human_mae_anchor_layer_ablation/anchor_all_blocks")}
    class X: epochs=50
    if not anchor.is_run_complete(ssl["pretrained_anchor"],anchor.mae_expected(source,.01,X(),0,ALL_BLOCKS),50): raise RuntimeError("Incomplete pretrained-anchor baseline")
    new=[]; reused=["pretrained_anchor Human SSL"]
    for name,value in CAT_RUNS:
        root=a.runs_dir/f"lambda_cat_{str(value).replace('.','p')}"/"seed0"; ssl[name]=root
        complete=(anchor.is_run_complete(root,expected(source,value),50) and
                  (root/"cat_train_feature_drift.csv").is_file())
        if a.force or not complete:
            run(ssl_command(a,source,name,value,root),f"Human MAE Cat-aware anchor {name}")
            if not (anchor.is_run_complete(root,expected(source,value),50) and
                    (root/"cat_train_feature_drift.csv").is_file()):
                raise RuntimeError(f"Incomplete Cat-aware Human SSL run: {root}")
            new.append(name+" Human SSL")
        else: reused.append(name+" Human SSL")
    probes={}
    for name,root in ssl.items():
        probe=root/"human_frozen_probe"; probes[name]=probe
        probe_ok=anchor.probe_complete(probe,anchor.probe_expected("human_mae",0))
        if a.force or not probe_ok:
            run(probe_command(a,source,root/"last_encoder.pt",probe,root/"config.json"),f"Human frozen probe {name}")
            if not anchor.probe_complete(probe,anchor.probe_expected("human_mae",0)):
                raise RuntimeError(f"Incomplete Human probe: {probe}")
            new.append(name+" Human probe")
        else: reused.append(name+" Human probe")
    controls={"imagenet":None,"full_human_mae":catrun.validate_human_ssl(catrun.full_ssl(0),0,False),"pretrained_anchor":ssl["pretrained_anchor"]/"last_encoder.pt"}
    checkpoints={**controls,**{name:root/"last_encoder.pt" for name,root in ssl.items() if name!="pretrained_anchor"}}
    cat_roots={}
    for method,checkpoint in checkpoints.items():
        # Existing cross-species controls are reused from their separate family when exact.
        old_name={"imagenet":"imagenet","full_human_mae":"human_mae_full","pretrained_anchor":"human_mae_anchor_all"}.get(method)
        old_root=catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),old_name,0) if old_name else None
        expected_cat=catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),old_name or method,0,None if checkpoint is None else Path(checkpoint).resolve())
        if old_root and catrun.cat_complete(old_root,expected_cat): root=old_root; reused.append(method+" Cat")
        else:
            root=cat_location(a.cat_runs_dir,method); run(cat_command(a,method,checkpoint),f"Cat full FT {method}")
            if not catrun.cat_complete(root,expected_cat): raise RuntimeError(f"Incomplete Cat run: {root}")
            new.append(method+" Cat")
        cat_roots[method]=root
    diagnostics=a.runs_dir/"diagnostics"; diagnostics.mkdir(exist_ok=True)
    summary=[]
    for method in ("imagenet","full_human_mae","pretrained_anchor",*[x[0] for x in CAT_RUNS]):
        checkpoint=checkpoints[method]; ssl_root=None if method in ("imagenet","full_human_mae") else ssl[method]
        if checkpoint is None:
            ssl_mae=human_loss=cat_loss=human_drift=cat_drift=""
            hdice,hiou,hloss=best_human(Path("runs/human_ssl_trajectory_probe/epoch_000"))
        else:
            human_path=(ssl_root/"final_feature_drift.csv" if ssl_root else diagnostics/f"{method}_human.csv"); cat_path=(ssl_root/"cat_train_feature_drift.csv" if ssl_root and (ssl_root/"cat_train_feature_drift.csv").is_file() else diagnostics/f"{method}_cat.csv")
            diagnostic(a,source,checkpoint,human_path,cat_path); human_drift=mean_drift(human_path); cat_drift=mean_drift(cat_path)
            ssl_mae=human_loss=cat_loss=""; hdice=hiou=hloss=""
            if ssl_root:
                last=rows(ssl_root/"ssl_metrics.csv")[-1]; ssl_mae=last["validation_mae_loss"]; human_loss=last["validation_feature_preserve_loss"]; cat_loss=last.get("validation_cat_anchor_loss",0); hdice,hiou,hloss=best_human(probes[method])
            else:
                full_root=catrun.full_ssl(0); metric_path=full_root/("ssl_metrics.csv" if (full_root/"ssl_metrics.csv").is_file() else "metrics.csv")
                ssl_mae=rows(metric_path)[-1]["validation_mae_loss"]
                hdice,hiou,hloss=best_human(full_root/"human_frozen_probe")
        cdice,ciou,closs=best_cat(cat_roots[method]); lc=dict(CAT_RUNS).get(method,0 if method=="pretrained_anchor" else "")
        summary.append({"method":method,"lambda_pretrained":0.01 if method not in ("imagenet","full_human_mae") else "","lambda_cat":lc,"ssl_val_mae_loss":ssl_mae,"ssl_val_human_anchor_loss":human_loss,"ssl_val_cat_anchor_loss":cat_loss,"mean_human_feature_drift":human_drift,"mean_cat_train_feature_drift":cat_drift,"human_frozen_dice":hdice,"cat_val_dice":cdice,"cat_val_iou":ciou,"cat_val_loss":closs,"checkpoint":"imagenet_pretrained" if checkpoint is None else str(Path(checkpoint).resolve())})
    write_csv(a.results_dir/"summary.csv",summary); by={r["method"]:r for r in summary}; deltas=[]
    for method,value in CAT_RUNS:
        r=by[method]; deltas.append({"method":method,"lambda_cat":value,"dice_minus_full_human_mae":r["cat_val_dice"]-by["full_human_mae"]["cat_val_dice"],"dice_minus_pretrained_anchor":r["cat_val_dice"]-by["pretrained_anchor"]["cat_val_dice"],"dice_minus_imagenet":r["cat_val_dice"]-by["imagenet"]["cat_val_dice"],"cat_drift_minus_pretrained_anchor":r["mean_cat_train_feature_drift"]-by["pretrained_anchor"]["mean_cat_train_feature_drift"],"human_drift_minus_pretrained_anchor":r["mean_human_feature_drift"]-by["pretrained_anchor"]["mean_human_feature_drift"]})
    write_csv(a.results_dir/"cat_aware_deltas.csv",deltas)
    selected=[by["pretrained_anchor"],*[by[n] for n,_ in CAT_RUNS]]; x=[float(r["lambda_cat"]) for r in selected]
    fig,ax=plt.subplots(); ax.plot(x,[r["cat_val_dice"] for r in selected],marker="o"); [ax.axhline(by[n]["cat_val_dice"],linestyle="--",label=n) for n in ("imagenet","full_human_mae","pretrained_anchor")]; ax.set(xlabel="lambda_cat",ylabel="Cat validation Dice"); ax.legend(); fig.tight_layout(); fig.savefig(a.results_dir/"lambda_cat_vs_cat_dice.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(); ax.plot(x,[r["mean_cat_train_feature_drift"] for r in selected],marker="o"); ax.set(xlabel="lambda_cat",ylabel="Mean Cat-train feature drift"); fig.tight_layout(); fig.savefig(a.results_dir/"lambda_cat_vs_cat_drift.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(); ax.scatter([r["mean_cat_train_feature_drift"] for r in selected],[r["cat_val_dice"] for r in selected]); ax.set(xlabel="Mean Cat-train feature drift",ylabel="Cat validation Dice"); fig.tight_layout(); fig.savefig(a.results_dir/"cat_drift_vs_cat_dice.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(); ax.plot(x,[r["mean_human_feature_drift"] for r in selected],marker="o",label="Human"); ax.plot(x,[r["mean_cat_train_feature_drift"] for r in selected],marker="o",label="Cat train"); ax.set(xlabel="lambda_cat",ylabel="Mean feature drift"); ax.legend(); fig.tight_layout(); fig.savefig(a.results_dir/"human_vs_cat_drift.png",dpi=200); plt.close(fig)
    print("New runs: "+", ".join(new)); print("Reused runs: "+", ".join(reused))


if __name__=="__main__": main()
