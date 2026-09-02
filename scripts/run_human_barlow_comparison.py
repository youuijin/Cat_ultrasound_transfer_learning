"""Train Human Barlow seed0 and compare downstream utility with ImageNet and Human MAE."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_cat_cross_species_anchor_validation as catrun
from scripts import run_human_mae_feature_anchor as anchor
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--python",default=sys.executable);p.add_argument("--barlow-run",type=Path,default=Path("runs/human_barlow/full/seed0"));p.add_argument("--results-dir",type=Path,default=Path("results/human_ssl_method_comparison"));p.add_argument("--cat-data-root",type=Path,default=Path("data/cat_dataset"));p.add_argument("--num-workers",type=int,default=4);p.add_argument("--force",action="store_true");p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);return p.parse_args()
def rows(p):return list(csv.DictReader(Path(p).open(encoding="utf-8")))
def write(p,data):
 with Path(p).open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def execute(cmd,label):print(f"\n{'='*72}\n{label}\n{'='*72}\n{subprocess.list2cmdline(cmd)}",flush=True);subprocess.run(cmd,cwd=ROOT,check=True)
def complete(root):
 root=Path(root);needed=(root/"config.json",root/"metrics.csv",root/"last_encoder.pt",root/"best_encoder.pt")
 if not all(x.is_file() for x in needed):return False
 c=json.loads((root/"config.json").read_text(encoding="utf-8"));r=rows(root/"metrics.csv");p=torch.load(root/"last_encoder.pt",map_location="cpu",weights_only=False)
 return c.get("ssl_method")=="barlow" and c.get("transfer")=="full" and c.get("seed")==0 and c.get("epochs")==50 and c.get("barlow_projector_dim")==2048 and c.get("barlow_lambda_offdiag")==.005 and len(r)==50 and int(r[-1]["epoch"])==49 and p.get("epoch")==49 and p.get("adaptation")=="human_kidney_ultrasound_barlow"
def best_human(root):
 r=max((x for x in rows(root/"metrics.csv") if x["phase"]=="validation"),key=lambda x:float(x["mean_dice"]));return float(r["mean_dice"]),float(r["kidney_iou"]),float(r["loss"])
def best_cat(root):
 r=max(rows(root/"metrics.csv"),key=lambda x:float(x["validation_mean_foreground_dice"]));return float(r["validation_mean_foreground_dice"]),float(r["validation_mean_foreground_iou"]),float(r["validation_loss"])
def main():
 a=parse();a.results_dir.mkdir(parents=True,exist_ok=True);source=json.loads(Path("checkpoints/human_mae_vit_b16_trajectory/config.json").read_text(encoding="utf-8"))
 if a.force or not complete(a.barlow_run):
  cmd=[a.python,"-m","src.human_ssl.train_barlow","--human1-root",source["human1_root"],"--human2-root",source["human2_root"],"--human3-root",source["human3_root"],"--val-fraction",str(source["val_fraction"]),"--batch-size","32","--epochs","50","--lr",str(source["lr"]),"--weight-decay",str(source["weight_decay"]),"--warmup-epochs",str(source["warmup_epochs"]),"--num-workers",str(a.num_workers),"--seed","0","--barlow-projector-dim","2048","--barlow-lambda-offdiag","0.005","--output-dir",str(a.barlow_run),"--amp" if a.amp else "--no-amp"];execute(cmd,"Human Barlow Twins full seed0")
 if not complete(a.barlow_run):raise RuntimeError("Incomplete Human Barlow run")
 checkpoint=(a.barlow_run/"last_encoder.pt").resolve();human_probe=a.barlow_run/"human_frozen_probe"
 if a.force or not anchor.probe_complete(human_probe,anchor.probe_expected("human_barlow",0)):
  cmd=[a.python,"-m","src.train_human_segmentation","--dataset","human2","--data-root",source["human2_root"],"--encoder","vit_b16","--encoder-init","human_barlow","--encoder-checkpoint",str(checkpoint),"--transfer","frozen","--val-fraction","0.2","--split-seed","42","--seed","0","--batch-size","8","--epochs","50","--lr","1e-4","--weight-decay","1e-4","--num-workers",str(a.num_workers),"--run-dir",str(human_probe),"--ssl-reference-config",str((a.barlow_run/"config.json").resolve()),"--amp" if a.amp else "--no-amp"];execute(cmd,"Human frozen probe Barlow")
 if not anchor.probe_complete(human_probe,anchor.probe_expected("human_barlow",0)):raise RuntimeError("Incomplete Barlow Human probe")
 cat_base=Path("runs/human_barlow/cat_downstream");cat_root=cat_base/"segmentation"/"vit_b16"/"full"/"fold_0"/"seed_0"/"init_human_barlow";barlow_cat_expected=catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),"human_barlow",0,checkpoint);barlow_cat_expected["encoder_init"]="human_barlow"
 if a.force or not catrun.cat_complete(cat_root,barlow_cat_expected):
  cmd=[a.python,"-m","src.segmentation.train","--encoder","vit_b16","--encoder-init","human_barlow","--encoder-checkpoint",str(checkpoint),"--transfer","full","--data-root",str(a.cat_data_root),"--num-folds","5","--fold","0","--split-seed","42","--seed","0","--batch-size","8","--epochs","50","--lr","1e-4","--weight-decay","1e-4","--num-workers",str(a.num_workers),"--output-dir",str(cat_base),"--amp" if a.amp else "--no-amp"];execute(cmd,"Cat segmentation Barlow seed0")
 if not catrun.cat_complete(cat_root,barlow_cat_expected):raise RuntimeError("Incomplete Barlow Cat run")
 image_probe=anchor.repro_paths(0)["imagenet"];full_root=catrun.full_ssl(0);full_probe=anchor.repro_paths(0)["full_probe"];image_cat=catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),"imagenet",0);full_cat=catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),"human_mae_full",0);full_checkpoint=catrun.validate_human_ssl(full_root,0,False)
 if not catrun.cat_complete(image_cat,catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),"imagenet",0,None)):raise RuntimeError("Incomplete ImageNet Cat control")
 if not catrun.cat_complete(full_cat,catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),"human_mae_full",0,full_checkpoint)):raise RuntimeError("Incomplete Full MAE Cat control")
 idice,iiou,iloss=best_human(image_probe);mdice,miou,mloss=best_human(full_probe);icdice,iciou,icloss=best_cat(image_cat);mcdice,mciou,mcloss=best_cat(full_cat);mae_metrics=rows(full_root/("ssl_metrics.csv" if (full_root/"ssl_metrics.csv").is_file() else "metrics.csv"))[-1];last=rows(a.barlow_run/"metrics.csv")[-1];hd,hi,hl=best_human(human_probe);cd,ci,cl=best_cat(cat_root)
 summary=[{"ssl_method":"imagenet","ssl_objective":"none","encoder_initialization":"ImageNet","ssl_seed":"","ssl_final_train_loss":"","ssl_final_val_loss":"","human_frozen_dice":idice,"human_frozen_iou":iiou,"human_frozen_loss":iloss,"cat_fold":0,"cat_seed":0,"cat_val_dice":icdice,"cat_val_iou":iciou,"cat_val_loss":icloss,"encoder_checkpoint":"imagenet_pretrained"},{"ssl_method":"human_mae_full","ssl_objective":"masked reconstruction","encoder_initialization":"ImageNet","ssl_seed":0,"ssl_final_train_loss":float(mae_metrics["train_mae_loss"]),"ssl_final_val_loss":float(mae_metrics["validation_mae_loss"]),"human_frozen_dice":mdice,"human_frozen_iou":miou,"human_frozen_loss":mloss,"cat_fold":0,"cat_seed":0,"cat_val_dice":mcdice,"cat_val_iou":mciou,"cat_val_loss":mcloss,"encoder_checkpoint":str(full_checkpoint)},{"ssl_method":"human_barlow_full","ssl_objective":"Barlow Twins","encoder_initialization":"ImageNet","ssl_seed":0,"ssl_final_train_loss":float(last["train_loss"]),"ssl_final_val_loss":float(last["validation_loss"]),"human_frozen_dice":hd,"human_frozen_iou":hi,"human_frozen_loss":hl,"cat_fold":0,"cat_seed":0,"cat_val_dice":cd,"cat_val_iou":ci,"cat_val_loss":cl,"encoder_checkpoint":str(checkpoint)}]
 write(a.results_dir/"seed0_summary.csv",summary);idx={r["ssl_method"]:r for r in summary};deltas=[]
 for method in ("human_mae_full","human_barlow_full"):
  r=idx[method];deltas.append({"method":method,"human_dice_minus_imagenet":r["human_frozen_dice"]-idx["imagenet"]["human_frozen_dice"],"cat_dice_minus_imagenet":r["cat_val_dice"]-idx["imagenet"]["cat_val_dice"],"human_dice_minus_mae":"" if method=="human_mae_full" else r["human_frozen_dice"]-idx["human_mae_full"]["human_frozen_dice"],"cat_dice_minus_mae":"" if method=="human_mae_full" else r["cat_val_dice"]-idx["human_mae_full"]["cat_val_dice"]})
 write(a.results_dir/"seed0_deltas.csv",deltas);labels=("ImageNet","Human MAE","Human Barlow")
 for field,name,ylabel in (("human_frozen_dice","ssl_method_vs_human_dice.png","Human frozen Dice"),("cat_val_dice","ssl_method_vs_cat_dice.png","Cat validation Dice")):
  fig,ax=plt.subplots();ax.plot(labels,[r[field] for r in summary],marker="o");ax.set(ylabel=ylabel);fig.tight_layout();fig.savefig(a.results_dir/name,dpi=200);plt.close(fig)
 fig,ax=plt.subplots();ax.scatter([r["human_frozen_dice"] for r in summary],[r["cat_val_dice"] for r in summary]);[ax.annotate(l,(r["human_frozen_dice"],r["cat_val_dice"])) for l,r in zip(labels,summary)];ax.set(xlabel="Human frozen Dice",ylabel="Cat validation Dice");fig.tight_layout();fig.savefig(a.results_dir/"human_vs_cat_transfer.png",dpi=200);plt.close(fig)
if __name__=="__main__":main()
