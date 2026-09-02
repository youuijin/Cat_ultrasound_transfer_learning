"""Stage 2: one all-block anchored Human Barlow run and its two downstream evaluations."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_cat_cross_species_anchor_validation as catrun
from scripts import run_human_barlow_comparison as comparison
from scripts import run_human_mae_feature_anchor as anchor

def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--python",default=sys.executable);p.add_argument("--full-run",type=Path,default=Path("runs/human_barlow/full/seed0"));p.add_argument("--anchored-run",type=Path,default=Path("runs/human_barlow/anchored_all_blocks/seed0"));p.add_argument("--results-dir",type=Path,default=Path("results/barlow_anchor_feasibility"));p.add_argument("--cat-data-root",type=Path,default=Path("data/cat_dataset"));p.add_argument("--num-workers",type=int,default=4);p.add_argument("--force",action="store_true");p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);return p.parse_args()
def rows(p):return list(csv.DictReader(Path(p).open(encoding="utf-8")))
def write(p,data):
 with Path(p).open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def execute(c,label):print(f"\n{'='*72}\n{label}\n{'='*72}\n{subprocess.list2cmdline(c)}",flush=True);subprocess.run(c,cwd=ROOT,check=True)
def complete(root,anchored):
 root=Path(root)
 if not all((root/x).is_file() for x in ("config.json","metrics.csv","last_encoder.pt","best_encoder.pt")):return False
 c=json.loads((root/"config.json").read_text(encoding="utf-8"));r=rows(root/"metrics.csv");p=torch.load(root/"last_encoder.pt",map_location="cpu",weights_only=False);expected=.01 if anchored else 0.
 return c.get("ssl_method")=="barlow" and c.get("transfer")=="full" and c.get("seed")==0 and c.get("epochs")==50 and c.get("barlow_projector_dim")==2048 and c.get("barlow_lambda_offdiag")==.005 and float(c.get("feature_anchor_lambda",0))==expected and (not anchored or c.get("feature_anchor_layers")==list(range(12))) and len(r)==50 and int(r[-1]["epoch"])==49 and p.get("epoch")==49 and p.get("adaptation")=="human_kidney_ultrasound_barlow" and float(p.get("config",{}).get("feature_anchor_lambda",0))==expected
def best_human(root):
 r=max((x for x in rows(Path(root)/"metrics.csv") if x["phase"]=="validation"),key=lambda x:float(x["mean_dice"]));return float(r["mean_dice"]),float(r["kidney_iou"])
def best_cat(root):
 r=max(rows(Path(root)/"metrics.csv"),key=lambda x:float(x["validation_mean_foreground_dice"]));return float(r["validation_mean_foreground_dice"]),float(r["validation_mean_foreground_iou"])
def main():
 a=parse();a.results_dir.mkdir(parents=True,exist_ok=True)
 if not complete(a.full_run,False):raise RuntimeError(f"Incomplete existing Full Human Barlow run: {a.full_run}")
 source=json.loads(Path("checkpoints/human_mae_vit_b16_trajectory/config.json").read_text(encoding="utf-8"))
 if a.force or not complete(a.anchored_run,True):
  cmd=[a.python,"-m","src.human_ssl.train_barlow","--human1-root",source["human1_root"],"--human2-root",source["human2_root"],"--human3-root",source["human3_root"],"--val-fraction",str(source["val_fraction"]),"--batch-size","32","--epochs","50","--lr",str(source["lr"]),"--weight-decay",str(source["weight_decay"]),"--warmup-epochs",str(source["warmup_epochs"]),"--num-workers",str(a.num_workers),"--seed","0","--barlow-projector-dim","2048","--barlow-lambda-offdiag","0.005","--feature-anchor-lambda","0.01","--output-dir",str(a.anchored_run),"--amp" if a.amp else "--no-amp"];execute(cmd,"Anchored Human Barlow seed0")
 if not complete(a.anchored_run,True):raise RuntimeError("Incomplete anchored Barlow run")
 checkpoint=(a.anchored_run/"last_encoder.pt").resolve();probe=a.anchored_run/"human_frozen_probe";expected_probe=anchor.probe_expected("human_barlow",0)
 if a.force or not anchor.probe_complete(probe,expected_probe):
  cmd=[a.python,"-m","src.train_human_segmentation","--dataset","human2","--data-root",source["human2_root"],"--encoder","vit_b16","--encoder-init","human_barlow","--encoder-checkpoint",str(checkpoint),"--transfer","frozen","--val-fraction","0.2","--split-seed","42","--seed","0","--batch-size","8","--epochs","50","--lr","1e-4","--weight-decay","1e-4","--num-workers",str(a.num_workers),"--run-dir",str(probe),"--ssl-reference-config",str((a.anchored_run/"config.json").resolve()),"--amp" if a.amp else "--no-amp"];execute(cmd,"Anchored Barlow Human frozen probe")
 if not anchor.probe_complete(probe,expected_probe):raise RuntimeError("Incomplete anchored Barlow Human probe")
 cat_base=Path("runs/human_barlow/anchored_cat_downstream");cat_root=cat_base/"segmentation"/"vit_b16"/"full"/"fold_0"/"seed_0"/"init_human_barlow";expected_cat=catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),"human_barlow",0,checkpoint);expected_cat["encoder_init"]="human_barlow"
 if a.force or not catrun.cat_complete(cat_root,expected_cat):
  cmd=[a.python,"-m","src.segmentation.train","--encoder","vit_b16","--encoder-init","human_barlow","--encoder-checkpoint",str(checkpoint),"--transfer","full","--data-root",str(a.cat_data_root),"--num-folds","5","--fold","0","--split-seed","42","--seed","0","--batch-size","8","--epochs","50","--lr","1e-4","--weight-decay","1e-4","--num-workers",str(a.num_workers),"--output-dir",str(cat_base),"--amp" if a.amp else "--no-amp"];execute(cmd,"Anchored Barlow Cat segmentation")
 if not catrun.cat_complete(cat_root,expected_cat):raise RuntimeError("Incomplete anchored Barlow Cat run")
 image_probe=anchor.repro_paths(0)["imagenet"];image_cat=catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),"imagenet",0);full_probe=a.full_run/"human_frozen_probe";full_cat=Path("runs/human_barlow/cat_downstream/segmentation/vit_b16/full/fold_0/seed_0/init_human_barlow")
 full_checkpoint=(a.full_run/"last_encoder.pt").resolve();full_expected=catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),"human_barlow",0,full_checkpoint);full_expected["encoder_init"]="human_barlow"
 if not anchor.probe_complete(full_probe,expected_probe) or not catrun.cat_complete(full_cat,full_expected):raise RuntimeError("Incomplete existing Full Barlow downstream control")
 if not catrun.cat_complete(image_cat,catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),"imagenet",0,None)):raise RuntimeError("Incomplete existing ImageNet Cat control")
 summary=[]
 for method,hroot,croot,sroot,ckpt in (("imagenet",image_probe,image_cat,None,"imagenet_pretrained"),("full_human_barlow",full_probe,full_cat,a.full_run,full_checkpoint),("anchored_human_barlow",probe,cat_root,a.anchored_run,checkpoint)):
  hd,hi=best_human(hroot);cd,ci=best_cat(croot);last=rows(Path(sroot)/"metrics.csv")[-1] if sroot else None;summary.append({"method":method,"human_frozen_dice":hd,"human_frozen_iou":hi,"cat_val_dice":cd,"cat_val_iou":ci,"ssl_val_loss":"" if last is None else float(last["validation_loss"]),"feature_anchor_loss":"" if last is None else float(last.get("validation_feature_anchor_loss",0)),"encoder_checkpoint":str(ckpt)})
 write(a.results_dir/"summary.csv",summary);diagnostics=[]
 for method,root in (("full_human_barlow",a.full_run),("anchored_human_barlow",a.anchored_run)):
  for r in rows(root/"metrics.csv"):diagnostics.append({"method":method,"epoch":int(r["epoch"]),"train_loss":float(r["train_loss"]),"validation_loss":float(r["validation_loss"]),"validation_mean_diagonal_correlation":float(r["validation_mean_diagonal_correlation"]),"validation_mean_abs_offdiag_correlation":float(r["validation_mean_abs_offdiag_correlation"]),"validation_projector_feature_std_mean":float(r["validation_projector_feature_std_mean"]),"validation_projector_feature_std_min":float(r["validation_projector_feature_std_min"])})
 write(a.results_dir/"collapse_diagnostics.csv",diagnostics);print(f"Anchored Barlow checkpoint: {checkpoint}");print(f"Summary: {(a.results_dir/'summary.csv').resolve()}");print(f"Collapse diagnostics: {(a.results_dir/'collapse_diagnostics.csv').resolve()}")
if __name__=="__main__":main()
