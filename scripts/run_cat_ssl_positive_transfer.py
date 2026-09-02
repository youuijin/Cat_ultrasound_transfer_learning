"""Feasibility comparison of Cat-only versus balanced Human+Cat MAE."""
from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_cat_cross_species_anchor_validation as catrun
from scripts import run_human_mae_feature_anchor as anchor

NEW=(("cat_only_mae_anchor","cat_only"),("human_cat_mixed_mae_anchor","human_cat"))

def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--python",default=sys.executable);p.add_argument("--runs-dir",type=Path,default=Path("runs/cat_ssl_positive_transfer"));p.add_argument("--results-dir",type=Path,default=Path("results/cat_ssl_positive_transfer"));p.add_argument("--cat-data-root",type=Path,default=Path("data/cat_dataset"));p.add_argument("--num-workers",type=int,default=4);p.add_argument("--force",action="store_true");p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);return p.parse_args()
def readj(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def readrows(p):return list(csv.DictReader(Path(p).open(encoding="utf-8")))
def number(value): return "" if value in ("",None) else float(value)
def write(p,data):
 with Path(p).open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def execute(cmd,label):print(f"\n{'='*72}\n{label}\n{'='*72}\n{subprocess.list2cmdline(cmd)}",flush=True);subprocess.run(cmd,cwd=ROOT,check=True)

def ssl_expected(source,domains):
 return {"ssl_domains":domains,"human1_root":source["human1_root"],"human2_root":source["human2_root"],"human3_root":source["human3_root"],"cat_data_root":str(Path("data/cat_dataset")),"val_fraction":source["val_fraction"],"mask_ratio":source["mask_ratio"],"norm_pixel_loss":source["norm_pixel_loss"],"decoder_dim":source["decoder_dim"],"decoder_depth":source["decoder_depth"],"decoder_heads":source["decoder_heads"],"batch_size":32,"epochs":50,"lr":source["lr"],"weight_decay":source["weight_decay"],"warmup_epochs":source["warmup_epochs"],"seed":0,"feature_anchor_lambda":.01,"feature_anchor_layers":list(range(12)),"cat_subject_overlap_count":0,"cat_labels_used":False,"cat_masks_used":False}
def ssl_complete(root,expected):
 root=Path(root); needed=(root/"config.json",root/"ssl_metrics.csv",root/"last_encoder.pt",root/"best_encoder.pt",root/"human_feature_drift.csv",root/"cat_train_feature_drift.csv",root/"cat_ssl_train_subjects.csv",root/"cat_downstream_val_subjects.csv")
 if not all(p.is_file() for p in needed):return False
 c=readj(root/"config.json")
 if any(c.get(k)!=v for k,v in expected.items()):return False
 r=readrows(root/"ssl_metrics.csv");p=torch.load(root/"last_encoder.pt",map_location="cpu",weights_only=False)
 return bool(r) and int(r[-1]["epoch"])==49 and int(r[-1]["optimizer_steps_completed"])==int(c["total_optimizer_steps"]) and p.get("epoch")==49 and p.get("ssl_domains")==expected["ssl_domains"]
def ssl_cmd(a,source,domains,root):
 return [a.python,"-m","src.human_ssl.train_domain_mae","--ssl-domains",domains,"--human1-root",source["human1_root"],"--human2-root",source["human2_root"],"--human3-root",source["human3_root"],"--cat-data-root",str(a.cat_data_root),"--val-fraction",str(source["val_fraction"]),"--mask-ratio",str(source["mask_ratio"]),"--no-norm-pixel-loss","--decoder-dim",str(source["decoder_dim"]),"--decoder-depth",str(source["decoder_depth"]),"--decoder-heads",str(source["decoder_heads"]),"--batch-size","32","--epochs","50","--lr",str(source["lr"]),"--weight-decay",str(source["weight_decay"]),"--warmup-epochs",str(source["warmup_epochs"]),"--num-workers",str(a.num_workers),"--seed","0","--feature-anchor-lambda","0.01","--output-dir",str(root),"--amp" if a.amp else "--no-amp"]
def probe_cmd(a,source,root,probe):
 class B:pass
 b=B();b.python=a.python;b.num_workers=a.num_workers;b.amp=a.amp
 return anchor.probe_command(b,source,root/"last_encoder.pt",probe,root/"config.json",0)
def best_human(root):
 r=max((x for x in readrows(root/"metrics.csv") if x["phase"]=="validation"),key=lambda x:float(x["mean_dice"]));return float(r["mean_dice"])
def best_cat(root):
 r=max(readrows(root/"metrics.csv"),key=lambda x:float(x["validation_mean_foreground_dice"]));return float(r["validation_mean_foreground_dice"]),float(r["validation_mean_foreground_iou"]),float(r["validation_loss"])
def drift(p):return sum(float(x["drift_1_minus_cka"]) for x in readrows(p))/4

def main():
 a=parse();a.runs_dir.mkdir(parents=True,exist_ok=True);a.results_dir.mkdir(parents=True,exist_ok=True);source=readj("checkpoints/human_mae_vit_b16_trajectory/config.json");new=[];reused=[];ssl={};probes={}
 for method,domains in NEW:
  root=a.runs_dir/method/"seed0";ssl[method]=root
  if a.force or not ssl_complete(root,ssl_expected(source,domains)):
   execute(ssl_cmd(a,source,domains,root),f"{method} SSL")
   if not ssl_complete(root,ssl_expected(source,domains)):raise RuntimeError(f"Incomplete SSL run {root}")
   new.append(method+" SSL")
  else:reused.append(method+" SSL")
  probe=root/"human_frozen_probe";probes[method]=probe
  if a.force or not anchor.probe_complete(probe,anchor.probe_expected("human_mae",0)):execute(probe_cmd(a,source,root,probe),f"{method} Human probe")
  if not anchor.probe_complete(probe,anchor.probe_expected("human_mae",0)):raise RuntimeError(f"Incomplete probe {probe}")
 cat_roots={}
 for method,_ in NEW:
  checkpoint=(ssl[method]/"last_encoder.pt").resolve();root=a.runs_dir/"cat_downstream"/method/"segmentation"/"vit_b16"/"full"/"fold_0"/"seed_0"/"init_human_mae";cat_roots[method]=root
  expected=catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),method,0,checkpoint)
  if a.force or not catrun.cat_complete(root,expected):
   cmd=[a.python,"-m","src.segmentation.train","--encoder","vit_b16","--encoder-init","human_mae","--encoder-checkpoint",str(checkpoint),"--transfer","full","--data-root",str(a.cat_data_root),"--num-folds","5","--fold","0","--split-seed","42","--seed","0","--batch-size","8","--epochs","50","--lr","1e-4","--weight-decay","1e-4","--num-workers",str(a.num_workers),"--output-dir",str(a.runs_dir/"cat_downstream"/method),"--amp" if a.amp else "--no-amp"];execute(cmd,f"{method} Cat segmentation");new.append(method+" Cat")
  else:reused.append(method+" Cat")
 full_checkpoint=catrun.validate_human_ssl(catrun.full_ssl(0),0,False);pretrained_checkpoint=catrun.validate_human_ssl(catrun.anchor_ssl(0),0,True);preservation_checkpoint=Path("runs/human_mae_cat_aware_anchor/lambda_cat_0p03/seed0/last_encoder.pt").resolve()
 control_specs=(("imagenet",catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),"imagenet",0),None),("full_human_mae",catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),"human_mae_full",0),full_checkpoint),("human_pretrained_anchor",catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),"human_mae_anchor_all",0),pretrained_checkpoint),("human_cat_preservation_anchor",Path("runs/cat_cat_aware_anchor_validation/cat_anchor_0p03/segmentation/vit_b16/full/fold_0/seed_0/init_human_mae"),preservation_checkpoint))
 for method,root,checkpoint in control_specs:
  expected=catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),"imagenet" if method=="imagenet" else method,0,checkpoint)
  if not catrun.cat_complete(root,expected):raise RuntimeError(f"Existing control is incomplete or mismatched: {method} -> {root}")
 controls=readrows("results/cat_aware_anchor_feasibility/summary.csv");byold={r["method"]:r for r in controls};mapping=(("imagenet","imagenet"),("full_human_mae","full_human_mae"),("human_pretrained_anchor","pretrained_anchor"),("human_cat_preservation_anchor","cat_anchor_0p03"));summary=[]
 for method,old in mapping:
  r=byold[old];summary.append({"method":method,"ssl_domains":"","human_cat_ssl_weight":"","ssl_optimizer_steps":"","ssl_val_mae_loss_human":number(r["ssl_val_mae_loss"]),"ssl_val_mae_loss_cat":"","mean_human_feature_drift":number(r["mean_human_feature_drift"]),"mean_cat_train_feature_drift":number(r["mean_cat_train_feature_drift"]),"human_frozen_dice":number(r["human_frozen_dice"]),"cat_val_dice":number(r["cat_val_dice"]),"cat_val_iou":number(r["cat_val_iou"]),"cat_val_loss":number(r["cat_val_loss"]),"encoder_checkpoint":r["checkpoint"]});reused.append(method+" control")
 for method,domains in NEW:
  root=ssl[method];last=readrows(root/"ssl_metrics.csv")[-1];cfg=readj(root/"config.json");d,i,l=best_cat(cat_roots[method]);summary.append({"method":method,"ssl_domains":"Cat" if domains=="cat_only" else "Human+Cat","human_cat_ssl_weight":0 if domains=="cat_only" else .5,"ssl_optimizer_steps":cfg["total_optimizer_steps"],"ssl_val_mae_loss_human":"" if domains=="cat_only" else last["validation_human_mae_loss"],"ssl_val_mae_loss_cat":last["validation_cat_mae_loss"],"mean_human_feature_drift":drift(root/"human_feature_drift.csv"),"mean_cat_train_feature_drift":drift(root/"cat_train_feature_drift.csv"),"human_frozen_dice":best_human(probes[method]),"cat_val_dice":d,"cat_val_iou":i,"cat_val_loss":l,"encoder_checkpoint":str((root/"last_encoder.pt").resolve())})
 write(a.results_dir/"feasibility_summary.csv",summary);idx={r["method"]:r for r in summary};mixed=idx["human_cat_mixed_mae_anchor"];delta=[{"cat_only_minus_imagenet":idx["cat_only_mae_anchor"]["cat_val_dice"]-idx["imagenet"]["cat_val_dice"],"mixed_minus_imagenet":mixed["cat_val_dice"]-idx["imagenet"]["cat_val_dice"],"mixed_minus_cat_only":mixed["cat_val_dice"]-idx["cat_only_mae_anchor"]["cat_val_dice"],"mixed_minus_human_pretrained_anchor":mixed["cat_val_dice"]-idx["human_pretrained_anchor"]["cat_val_dice"],"mixed_minus_human_cat_preservation_anchor":mixed["cat_val_dice"]-idx["human_cat_preservation_anchor"]["cat_val_dice"]}];write(a.results_dir/"positive_transfer_deltas.csv",delta)
 labels=("ImageNet","Human MAE","Human preserve","Human+Cat preserve","Cat-only MAE","Human+Cat MAE");fig,ax=plt.subplots();ax.plot(labels,[r["cat_val_dice"] for r in summary],marker="o");ax.set(ylabel="Cat validation Dice");fig.autofmt_xdate(rotation=20);fig.tight_layout();fig.savefig(a.results_dir/"ssl_source_vs_cat_dice.png",dpi=200);plt.close(fig)
 fig,ax=plt.subplots();small=(idx["cat_only_mae_anchor"],idx["human_cat_mixed_mae_anchor"],idx["imagenet"]);ax.bar(("Cat-only MAE","Human+Cat MAE","ImageNet"),[r["cat_val_dice"] for r in small]);ax.set(ylabel="Cat validation Dice");fig.tight_layout();fig.savefig(a.results_dir/"cat_only_vs_mixed.png",dpi=200);plt.close(fig)
 relevant=[r for r in summary if r["mean_cat_train_feature_drift"] not in ("",None)];fig,ax=plt.subplots();ax.scatter([float(r["mean_cat_train_feature_drift"]) for r in relevant],[float(r["cat_val_dice"]) for r in relevant]);[ax.annotate(r["method"],(float(r["mean_cat_train_feature_drift"]),float(r["cat_val_dice"]))) for r in relevant];ax.set(xlabel="Mean Cat-train feature drift",ylabel="Cat validation Dice");fig.tight_layout();fig.savefig(a.results_dir/"feature_drift_vs_cat_dice.png",dpi=200);plt.close(fig)
 print("New runs: "+", ".join(new));print("Reused controls: "+", ".join(reused))
if __name__=="__main__":main()
