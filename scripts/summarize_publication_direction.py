"""Combine completed Stage 1 and Stage 2 outputs without running training."""
from __future__ import annotations
import csv
from pathlib import Path

def rows(p):return list(csv.DictReader(Path(p).open(encoding="utf-8")))
def main():
 stage1=rows("results/constrained_human_mae_cat_transfer/all_seed_results.csv");barlow=rows("results/barlow_anchor_feasibility/summary.csv");human=rows("results/human_mae_adaptation_depth_reproducibility/all_depth_seed_results.csv");human_idx={(r["adaptation_depth"],int(r["seed"])):r for r in human};image={int(r["cat_seed"]):float(r["cat_val_dice"]) for r in stage1 if r["method"]=="imagenet"};out=[]
 labels={"imagenet":"ImageNet","full_human_mae":"Full Human MAE","alpha0p1":"alpha0p1 Human MAE","last2":"last2 Human MAE","last4":"last4 Human MAE","last6":"last6 Human MAE"}
 for r in stage1:
  method=r["method"];seed=int(r["cat_seed"]);depth={"imagenet":"imagenet","full_human_mae":"full","last2":"last2","last4":"last4","last6":"last6"}.get(method);hf=human_idx.get((depth,seed),{}).get("human_frozen_dice","") if depth else "";out.append({"experiment_family":"Human MAE constrained transfer","method":labels[method],"seed":seed,"human_frozen_dice":hf,"cat_val_dice":r["cat_val_dice"],"cat_delta_vs_imagenet":float(r["cat_val_dice"])-image[seed],"notes":"Cat fold0; seed-matched"})
 bimage=float(next(r for r in barlow if r["method"]=="imagenet")["cat_val_dice"])
 for r in barlow:
  out.append({"experiment_family":"Human Barlow preservation","method":{"imagenet":"ImageNet","full_human_barlow":"Full Human Barlow","anchored_human_barlow":"Anchored Human Barlow"}[r["method"]],"seed":0,"human_frozen_dice":r["human_frozen_dice"],"cat_val_dice":r["cat_val_dice"],"cat_delta_vs_imagenet":float(r["cat_val_dice"])-bimage,"notes":"Cat fold0; all-block anchor lambda=0.01" if r["method"]=="anchored_human_barlow" else "Cat fold0"})
 path=Path("results/publication_direction_summary.csv");path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 summary=rows("results/constrained_human_mae_cat_transfer/method_summary.csv");delta={r["method"]:r for r in rows("results/constrained_human_mae_cat_transfer/vs_imagenet_summary.csv")};best=max((r for r in summary if r["method"] in ("last2","last4","last6")),key=lambda r:float(r["mean_dice"]));print("STAGE 1\n------------------------------------------------\nMethod      Mean Cat Dice   SD   Delta vs ImageNet")
 for r in summary:print(f"{r['method']:<12}{float(r['mean_dice']):.6f}        {float(r['std_dice']):.6f}   {float(delta.get(r['method'],{}).get('mean_delta_dice',0)):.6f}")
 print(f"\nBest constrained Human MAE:\n{best['method']} mean Dice={float(best['mean_dice']):.6f}")
 print("\nSTAGE 2\n------------------------------------------------")
 for r in barlow:print(f"{r['method']}: Human Dice={float(r['human_frozen_dice']):.6f} Cat Dice={float(r['cat_val_dice']):.6f}")
 print("\nOutput paths:\nresults/constrained_human_mae_cat_transfer\nresults/barlow_anchor_feasibility\nresults/publication_direction_summary.csv\nresults/constrained_human_mae_cat_transfer/constrained_human_mae_cat_transfer.png")
if __name__=="__main__":main()
