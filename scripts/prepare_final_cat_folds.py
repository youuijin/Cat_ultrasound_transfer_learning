"""Prepare (print or write) final Cat commands; never executes them."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
METHODS=("imagenet","full_human_mae","last2","last4","last6","alpha0p1","full_human_barlow","anchored_human_barlow")
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--method",required=True,choices=METHODS);p.add_argument("--checkpoint",type=Path);p.add_argument("--fold",nargs="+",type=int,default=[0,1,2,3,4]);p.add_argument("--seed",nargs="+",type=int,default=[0,1,2]);p.add_argument("--python",default=sys.executable);p.add_argument("--data-root",type=Path,default=Path("data/cat_dataset"));p.add_argument("--output-dir",type=Path,default=Path("runs/final_cat_folds"));p.add_argument("--command-file",type=Path,default=Path("results/final_cat_folds_commands.txt"));p.add_argument("--num-workers",type=int,default=4);p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);a=p.parse_args()
 if any(x not in range(5) for x in a.fold):p.error("--fold values must be 0..4")
 if a.method!="imagenet" and a.checkpoint is None:p.error("non-ImageNet methods require --checkpoint")
 return a
def main():
 a=parse();init="imagenet" if a.method=="imagenet" else ("human_barlow" if "barlow" in a.method else "human_mae");commands=[]
 for fold in a.fold:
  for seed in a.seed:
   c=[a.python,"-m","src.segmentation.train","--encoder","vit_b16","--encoder-init",init,"--transfer","full","--data-root",str(a.data_root),"--num-folds","5","--fold",str(fold),"--split-seed","42","--seed",str(seed),"--batch-size","8","--epochs","50","--lr","1e-4","--weight-decay","1e-4","--num-workers",str(a.num_workers),"--output-dir",str(a.output_dir/a.method),"--amp" if a.amp else "--no-amp"]
   if a.checkpoint:c.extend(["--encoder-checkpoint",str(a.checkpoint.resolve())])
   commands.append(subprocess.list2cmdline(c))
 a.command_file.parent.mkdir(parents=True,exist_ok=True);a.command_file.write_text("\n".join(commands)+"\n",encoding="utf-8");print(f"Prepared {len(commands)} commands only; nothing executed.\n{a.command_file.resolve()}")
if __name__=="__main__":main()
