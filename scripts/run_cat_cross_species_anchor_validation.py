"""Run seed-matched Cat segmentation from ImageNet, Full Human MAE, and anchored MAE."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("imagenet", "human_mae_full", "human_mae_anchor_all")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=Path("data/cat_dataset"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/cat_cross_species_anchor_validation"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/cat_cross_species_anchor_validation"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.fold != 0 or sorted(args.seeds) != [0, 1, 2]:
        parser.error("This feasibility launcher is fixed to fold 0 and seeds 0 1 2")
    return args


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def full_ssl(seed):
    if seed == 0: return Path("runs/human_mae_recipe_ablation/baseline")
    return Path(f"runs/human_mae_adaptation_depth/full/seed{seed}/mae")


def anchor_ssl(seed):
    if seed == 0: return Path("runs/human_mae_anchor_layer_ablation/anchor_all_blocks")
    return Path(f"runs/human_mae_all_blocks_reproducibility/seed{seed}")


def validate_human_ssl(root, seed, anchored):
    root = Path(root); config_path = root / "config.json"; metrics_path = root / "ssl_metrics.csv"
    if not metrics_path.is_file(): metrics_path = root / "metrics.csv"
    required = [config_path, metrics_path, root / "last_encoder.pt", root / "best_encoder.pt"]
    if anchored: required.append(root / "final_feature_drift.csv")
    missing_files = [str(path) for path in required if not path.is_file()]
    if missing_files: raise RuntimeError("Incomplete Human SSL run; missing: " + ", ".join(missing_files))
    config = read_json(config_path); epochs = int(config["epochs"]); rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    final_epoch = int(rows[-1]["epoch"]) if rows else -1
    payload = torch.load(root / "last_encoder.pt", map_location="cpu", weights_only=False)
    checkpoint_config = payload.get("config", {})
    expected_layers = list(range(12)) if anchored else None
    errors = []
    if final_epoch != epochs - 1: errors.append(f"metrics final epoch {final_epoch} != {epochs - 1}")
    if payload.get("epoch") != epochs - 1: errors.append(f"checkpoint epoch {payload.get('epoch')} != {epochs - 1}")
    if payload.get("adaptation") != "human_kidney_ultrasound_mae": errors.append("wrong adaptation metadata")
    if int(config.get("seed", -1)) != seed or int(checkpoint_config.get("seed", -1)) != seed: errors.append("seed mismatch")
    if config.get("encoder_trainable_last_blocks") is not None: errors.append("encoder adaptation is not full")
    expected_lambda = 0.01 if anchored else 0.0
    if float(config.get("feature_anchor_lambda", 0.0)) != expected_lambda: errors.append("feature lambda mismatch")
    if anchored and config.get("feature_anchor_layers") != expected_layers: errors.append("anchor layers are not blocks 0-11")
    if float(checkpoint_config.get("feature_anchor_lambda", 0.0)) != expected_lambda: errors.append("checkpoint lambda mismatch")
    if anchored and checkpoint_config.get("feature_anchor_layers") != expected_layers: errors.append("checkpoint anchor-layer mismatch")
    if errors: raise RuntimeError(f"Invalid Human SSL checkpoint {root}: " + "; ".join(errors))
    print(f"Human checkpoint seed {seed}: {root.resolve()}")
    print(f"  epoch={payload['epoch']} adaptation={payload['adaptation']} lambda={expected_lambda} layers={expected_layers}")
    return (root / "last_encoder.pt").resolve()


def run_dir(args, method, seed):
    base = args.runs_dir / method / "segmentation" / "vit_b16" / "full" / f"fold_{args.fold}" / f"seed_{seed}"
    return base if method == "imagenet" else base / "init_human_mae"


def expected_config(args, method, seed, checkpoint):
    return {"task": "segmentation", "encoder": "vit_b16",
            "encoder_init": "imagenet" if method == "imagenet" else "human_mae",
            "encoder_checkpoint": None if checkpoint is None else str(checkpoint),
            "transfer": "full", "data_root": str(args.data_root), "num_folds": 5,
            "fold": args.fold, "split_seed": 42, "seed": seed, "batch_size": 8,
            "epochs": 50, "lr": 1e-4, "weight_decay": 1e-4, "amp": args.amp}


def cat_complete(root, expected, reference_split=None):
    root = Path(root); config_path=root/"config.json"; metrics_path=root/"metrics.csv"; last=root/"last.pt"
    if not all(path.is_file() for path in (config_path, metrics_path, last, root/"best.pt",
                                            root/"train_subjects.txt", root/"val_subjects.txt")):
        return False
    config=read_json(config_path)
    if any(config.get(key) != value for key,value in expected.items()): return False
    rows=list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    if not rows or int(rows[-1]["epoch"]) != expected["epochs"]-1: return False
    payload=torch.load(last,map_location="cpu",weights_only=False)
    if int(payload.get("epoch",-1)) != expected["epochs"]-1: return False
    split=(tuple(config["train_subjects"]),tuple(config["val_subjects"]))
    text_split=((root/"train_subjects.txt").read_text(encoding="utf-8").splitlines(),
                (root/"val_subjects.txt").read_text(encoding="utf-8").splitlines())
    if [Path(path).name for path in split[0]] != text_split[0] or [Path(path).name for path in split[1]] != text_split[1]: return False
    if reference_split is not None and split != reference_split: return False
    if expected["encoder_init"] in ("human_mae", "human_dino", "human_barlow"):
        summary=config.get("encoder_load_summary", {})
        if (summary.get("missing_keys") or summary.get("unexpected_keys") or
                summary.get("shape_mismatch_keys") or
                Path(summary.get("checkpoint", "")).resolve() != Path(expected["encoder_checkpoint"]).resolve()): return False
    return True


def command(args, method, seed, checkpoint):
    result=[args.python,"-m","src.segmentation.train","--encoder","vit_b16","--encoder-init",
            "imagenet" if method=="imagenet" else "human_mae","--transfer","full",
            "--data-root",str(args.data_root),"--num-folds","5","--fold",str(args.fold),
            "--split-seed","42","--seed",str(seed),"--batch-size","8","--epochs","50",
            "--lr","1e-4","--weight-decay","1e-4","--num-workers",str(args.num_workers),
            "--output-dir",str(args.runs_dir/method),"--amp" if args.amp else "--no-amp"]
    if checkpoint is not None: result.extend(["--encoder-checkpoint",str(checkpoint)])
    return result


def execute(cmd,label):
    print(f"\n{'='*72}\n{label}\n{'='*72}\n{subprocess.list2cmdline(cmd)}",flush=True)
    subprocess.run(cmd,cwd=ROOT,check=True)


def best_metrics(root):
    rows=list(csv.DictReader((root/"metrics.csv").open(encoding="utf-8")))
    best=max(rows,key=lambda row:float(row["validation_mean_foreground_dice"]))
    return int(best["epoch"]),float(best["validation_mean_foreground_dice"]),float(best["validation_mean_foreground_iou"]),float(best["validation_loss"])


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def mean_std(values): values=list(values); return statistics.mean(values),statistics.stdev(values)


def delta_summary(paired, include_iou=True):
    values=[row["delta_dice"] for row in paired]
    result={"mean_delta_dice":statistics.mean(values),"std_delta_dice":statistics.stdev(values),
            "positive_delta_count":sum(x>0 for x in values),"negative_delta_count":sum(x<0 for x in values),"total_seeds":len(values)}
    if include_iou:
        iou=[row["delta_iou"] for row in paired]; result.update(mean_delta_iou=statistics.mean(iou),std_delta_iou=statistics.stdev(iou))
    return result


def main():
    args=parse_args(); args.results_dir.mkdir(parents=True,exist_ok=True); checkpoints={}
    for seed in args.seeds:
        checkpoints[("human_mae_full",seed)]=validate_human_ssl(full_ssl(seed),seed,False)
        checkpoints[("human_mae_anchor_all",seed)]=validate_human_ssl(anchor_ssl(seed),seed,True)
    all_rows=[]; reference_splits={}; new_runs=[]; reused_runs=[]
    for seed in args.seeds:
        for method in METHODS:
            checkpoint=checkpoints.get((method,seed)); root=run_dir(args,method,seed); expected=expected_config(args,method,seed,checkpoint)
            valid=(not args.force and cat_complete(root,expected,reference_splits.get(seed)))
            if valid: reused_runs.append(f"{method}/seed{seed}"); print(f"[reuse] {root.resolve()}")
            else:
                execute(command(args,method,seed,checkpoint),f"Cat segmentation {method} seed {seed}")
                if not cat_complete(root,expected,reference_splits.get(seed)): raise RuntimeError(f"Incomplete or mismatched Cat run: {root}")
                new_runs.append(f"{method}/seed{seed}")
            config=read_json(root/"config.json"); split=(tuple(config["train_subjects"]),tuple(config["val_subjects"]))
            reference_splits.setdefault(seed,split)
            if split != reference_splits[seed]: raise RuntimeError(f"Cat subject split mismatch for seed {seed}")
            epoch,dice,iou,loss=best_metrics(root)
            all_rows.append({"initialization":method,"human_ssl_seed":"" if method=="imagenet" else seed,"cat_seed":seed,
                             "fold":args.fold,"encoder_checkpoint":"imagenet_pretrained" if checkpoint is None else str(checkpoint),
                             "best_epoch":epoch,"val_dice":dice,"val_iou":iou,"val_loss":loss})
    write_csv(args.results_dir/"all_runs.csv",all_rows); index={(r["initialization"],r["cat_seed"]):r for r in all_rows}
    summaries=[]
    for method in METHODS:
        selected=[r for r in all_rows if r["initialization"]==method]; dm,ds=mean_std(r["val_dice"] for r in selected); im,is_=mean_std(r["val_iou"] for r in selected); lm,ls=mean_std(r["val_loss"] for r in selected)
        summaries.append({"initialization":method,"n_runs":3,"mean_dice":dm,"std_dice":ds,"mean_iou":im,"std_iou":is_,"mean_loss":lm,"std_loss":ls})
    write_csv(args.results_dir/"method_summary.csv",summaries); comparisons={}
    for base,stem in (("human_mae_full","anchor_vs_full"),("imagenet","anchor_vs_imagenet")):
        paired=[]
        for seed in args.seeds:
            b=index[(base,seed)]; a=index[("human_mae_anchor_all",seed)]; prefix="full" if base=="human_mae_full" else "imagenet"
            paired.append({"seed":seed,f"{prefix}_dice":b["val_dice"],"anchor_dice":a["val_dice"],"delta_dice":a["val_dice"]-b["val_dice"],f"{prefix}_iou":b["val_iou"],"anchor_iou":a["val_iou"],"delta_iou":a["val_iou"]-b["val_iou"]})
        write_csv(args.results_dir/f"{stem}.csv",paired); write_csv(args.results_dir/f"{stem}_summary.csv",[delta_summary(paired)]); comparisons[base]=paired
    paired=[]
    for seed in args.seeds:
        image=index[("imagenet",seed)]; full=index[("human_mae_full",seed)]; paired.append({"seed":seed,"imagenet_dice":image["val_dice"],"full_dice":full["val_dice"],"delta_dice":full["val_dice"]-image["val_dice"]})
    write_csv(args.results_dir/"full_vs_imagenet.csv",paired); write_csv(args.results_dir/"full_vs_imagenet_summary.csv",[delta_summary(paired,False)])
    labels=("ImageNet","Human MAE","Anchored Human MAE")
    fig,ax=plt.subplots()
    for seed in args.seeds: ax.plot(labels,[index[(m,seed)]["val_dice"] for m in METHODS],marker="o",label=f"seed {seed}")
    ax.set(ylabel="Cat validation Dice"); ax.legend(); fig.tight_layout(); fig.savefig(args.results_dir/"cat_cross_species_seed_comparison.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots()
    for base,values in comparisons.items(): ax.plot(args.seeds,[r["delta_dice"] for r in values],marker="o",label=f"Anchor - {base}")
    ax.axhline(0,color="black",linestyle="--"); ax.set(xlabel="Seed",ylabel="Paired Dice delta",xticks=args.seeds); ax.legend(); fig.tight_layout(); fig.savefig(args.results_dir/"cat_anchor_paired_delta.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(); xs=range(3)
    ax.errorbar(xs,[r["mean_dice"] for r in summaries],yerr=[r["std_dice"] for r in summaries],fmt="o",capsize=4)
    for x,method in zip(xs,METHODS): ax.scatter([x]*3,[index[(method,s)]["val_dice"] for s in args.seeds])
    ax.set(xticks=list(xs),xticklabels=labels,ylabel="Cat validation Dice"); fig.tight_layout(); fig.savefig(args.results_dir/"cat_initialization_summary.png",dpi=200); plt.close(fig)
    print("Method                    Mean Dice     SD\n------------------------------------------------")
    for row in summaries: print(f"{row['initialization']:<25} {row['mean_dice']:.6f}  {row['std_dice']:.6f}")
    for label,values in (("Anchor - Full",comparisons["human_mae_full"]),("Anchor - ImageNet",comparisons["imagenet"]),("Full - ImageNet",paired)):
        print(label+": "+", ".join(f"seed{r['seed']}={r['delta_dice']:.6f}" for r in values)+f", mean={statistics.mean(r['delta_dice'] for r in values):.6f}")
    print("New runs: "+", ".join(new_runs)); print("Reused runs: "+", ".join(reused_runs))


if __name__ == "__main__": main()
