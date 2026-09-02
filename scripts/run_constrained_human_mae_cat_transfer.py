"""Stage 1: seed-matched constrained Human MAE checkpoints on Cat segmentation."""
from __future__ import annotations
import argparse,csv,json,statistics,subprocess,sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_cat_cross_species_anchor_validation as catrun
from scripts import run_human_mae_weight_interpolation as interp
METHODS=("imagenet","full_human_mae","last2","last4","last6","alpha0p1")
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--python",default=sys.executable);p.add_argument("--runs-dir",type=Path,default=Path("runs/constrained_human_mae_cat_transfer"));p.add_argument("--results-dir",type=Path,default=Path("results/constrained_human_mae_cat_transfer"));p.add_argument("--cat-data-root",type=Path,default=Path("data/cat_dataset"));p.add_argument("--num-workers",type=int,default=4);p.add_argument("--force",action="store_true");p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);return p.parse_args()
def rows(p):return list(csv.DictReader(Path(p).open(encoding="utf-8")))
def write(p,data):
 with Path(p).open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def execute(c,l):print(f"\n{'='*72}\n{l}\n{'='*72}\n{subprocess.list2cmdline(c)}",flush=True);subprocess.run(c,cwd=ROOT,check=True)
def depth_inventory():return {(r["adaptation_depth"],int(r["seed"])):Path(r["mae_checkpoint"]).resolve() for r in rows("results/human_mae_adaptation_depth_reproducibility/all_depth_seed_results.csv") if r["adaptation_depth"] in ("last2","last4","last6")}
def validate_depth(method,seed,path):
 root=path.parent;c=json.loads((root/"config.json").read_text(encoding="utf-8"));metric=root/("ssl_metrics.csv" if (root/"ssl_metrics.csv").is_file() else "metrics.csv");r=rows(metric);p=torch.load(path,map_location="cpu",weights_only=False);depth=int(method.removeprefix("last"));errors=[]
 if c.get("seed")!=seed:errors.append("seed")
 if c.get("encoder_trainable_last_blocks")!=depth:errors.append("adaptation depth")
 if int(r[-1]["epoch"])!=int(c["epochs"])-1:errors.append("metrics final epoch")
 if p.get("epoch")!=int(c["epochs"])-1:errors.append("checkpoint epoch")
 if p.get("adaptation")!="human_kidney_ultrasound_mae":errors.append("adaptation metadata")
 reference=torch.load(catrun.full_ssl(seed)/"last_encoder.pt",map_location="cpu",weights_only=False)["state_dict"];state=p.get("state_dict",{});missing=sorted(set(reference)-set(state));unexpected=sorted(set(state)-set(reference));shape_mismatch=sorted(k for k in set(reference)&set(state) if reference[k].shape!=state[k].shape)
 if missing:errors.append(f"missing parameters={len(missing)}")
 if unexpected:errors.append(f"unexpected parameters={len(unexpected)}")
 if shape_mismatch:errors.append(f"shape mismatches={len(shape_mismatch)}")
 if errors:raise RuntimeError(f"Invalid {method} seed{seed}: {errors}")
 print(f"method={method} Human SSL seed={seed} checkpoint={path} trainable_blocks=last {depth}; missing=0 unexpected=0 shape_mismatch=0")
def best(root):
 r=max(rows(root/"metrics.csv"),key=lambda x:float(x["validation_mean_foreground_dice"]));return float(r["validation_mean_foreground_dice"]),float(r["validation_mean_foreground_iou"]),float(r["validation_loss"])
def main():
 a=parse();a.runs_dir.mkdir(parents=True,exist_ok=True);a.results_dir.mkdir(parents=True,exist_ok=True);inventory=depth_inventory();checkpoints={}
 for seed in (0,1,2):
  checkpoints[("imagenet",seed)]=None;checkpoints[("full_human_mae",seed)]=Path(catrun.validate_human_ssl(catrun.full_ssl(seed),seed,False)).resolve()
  for method in ("last2","last4","last6"):validate_depth(method,seed,inventory[(method,seed)]);checkpoints[(method,seed)]=inventory[(method,seed)]
  checkpoints[("alpha0p1",seed)],_human,_diff,_relative=interp.generate_alpha_0p1(argparse.Namespace(runs_dir=Path("runs/human_mae_weight_interpolation"),repro_runs_dir=Path("runs/human_mae_alpha_0p1_reproducibility"),force=False),seed)
 records=[];new=[];reused=[]
 for seed in (0,1,2):
  for method in METHODS:
   checkpoint=checkpoints[(method,seed)]
   if method=="imagenet":root=catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),"imagenet",seed);expected_method="imagenet"
   elif method=="full_human_mae":root=catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),"human_mae_full",seed);expected_method="human_mae_full"
   elif method=="alpha0p1":root=interp.interpolation_cat_root(Path("runs/human_mae_weight_interpolation"),.1) if seed==0 else interp.repro_cat_root(Path("runs/human_mae_alpha_0p1_reproducibility"),seed);expected_method="alpha_0p1"
   else:root=a.runs_dir/method/f"seed{seed}"/"segmentation"/"vit_b16"/"full"/"fold_0"/f"seed_{seed}"/"init_human_mae";expected_method=method
   expected=catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),expected_method,seed,checkpoint);complete=catrun.cat_complete(root,expected)
   if method in ("imagenet","full_human_mae","alpha0p1") and not complete:raise RuntimeError(f"Incomplete reused control {method} seed{seed}")
   if method in ("last2","last4","last6") and (a.force or not complete):
    out=a.runs_dir/method/f"seed{seed}";cmd=[a.python,"-m","src.segmentation.train","--encoder","vit_b16","--encoder-init","human_mae","--encoder-checkpoint",str(checkpoint),"--transfer","full","--data-root",str(a.cat_data_root),"--num-folds","5","--fold","0","--split-seed","42","--seed",str(seed),"--batch-size","8","--epochs","50","--lr","1e-4","--weight-decay","1e-4","--num-workers",str(a.num_workers),"--output-dir",str(out),"--amp" if a.amp else "--no-amp"];execute(cmd,f"Cat {method} seed{seed}");new.append(f"{method}/seed{seed}")
    if not catrun.cat_complete(root,expected):raise RuntimeError(f"Incomplete Cat run {method} seed{seed}")
   else:reused.append(f"{method}/seed{seed}")
   d,i,l=best(root);records.append({"method":method,"human_ssl_seed":"" if method=="imagenet" else seed,"cat_seed":seed,"cat_fold":0,"cat_val_dice":d,"cat_val_iou":i,"cat_val_loss":l,"encoder_checkpoint":"imagenet_pretrained" if checkpoint is None else str(checkpoint),"reused_cat_run":complete and not a.force})
 write(a.results_dir/"all_seed_results.csv",records);idx={(r["method"],r["cat_seed"]):r for r in records};summary=[]
 for method in METHODS:
  s=[idx[(method,x)] for x in (0,1,2)];summary.append({"method":method,"n_seeds":3,"mean_dice":statistics.mean(r["cat_val_dice"] for r in s),"std_dice":statistics.stdev(r["cat_val_dice"] for r in s),"mean_iou":statistics.mean(r["cat_val_iou"] for r in s),"std_iou":statistics.stdev(r["cat_val_iou"] for r in s),"mean_loss":statistics.mean(r["cat_val_loss"] for r in s),"std_loss":statistics.stdev(r["cat_val_loss"] for r in s)})
 write(a.results_dir/"method_summary.csv",summary);paired=[];paired_summary=[]
 for method in ("full_human_mae","last2","last4","last6","alpha0p1"):
  vals=[]
  for seed in (0,1,2):
   image=idx[("imagenet",seed)]["cat_val_dice"];value=idx[(method,seed)]["cat_val_dice"];delta=value-image;vals.append(delta);paired.append({"method":method,"seed":seed,"imagenet_dice":image,"method_dice":value,"delta_dice":delta})
  paired_summary.append({"method":method,"mean_delta_dice":statistics.mean(vals),"std_delta_dice":statistics.stdev(vals),"positive_seed_count":sum(v>0 for v in vals),"negative_seed_count":sum(v<0 for v in vals),"total_seeds":3})
 write(a.results_dir/"vs_imagenet.csv",paired);write(a.results_dir/"vs_imagenet_summary.csv",paired_summary);fig,ax=plt.subplots();labels=("ImageNet","Full MAE","Last2","Last4","Last6","alpha=0.1")
 for seed in (0,1,2):ax.plot(labels,[idx[(m,seed)]["cat_val_dice"] for m in METHODS],marker="o",label=f"seed {seed}")
 ax.set(ylabel="Cat validation Dice");ax.legend();fig.tight_layout();fig.savefig(a.results_dir/"constrained_human_mae_cat_transfer.png",dpi=200);plt.close(fig);best_method=max((r for r in summary if r["method"] in ("last2","last4","last6")),key=lambda r:r["mean_dice"]);delta=next(r["mean_delta_dice"] for r in paired_summary if r["method"]==best_method["method"]);print(f"Best constrained Human MAE: {best_method['method']} mean Dice={best_method['mean_dice']:.6f} mean delta vs ImageNet={delta:.6f}");print("New Cat runs: "+", ".join(new));print("Reused Cat runs: "+", ".join(reused))
if __name__=="__main__":main()
