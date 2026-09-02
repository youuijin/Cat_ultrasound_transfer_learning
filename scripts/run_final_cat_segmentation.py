"""Final publication Cat segmentation: ImageNet, Full Human MAE, and Last2 Human MAE."""
from __future__ import annotations
import argparse,csv,json,statistics,subprocess,sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from scipy.stats import wilcoxon

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.classification.data import split_subjects

METHODS=("imagenet","full_human_mae","constrained_human_mae_last2")
DISPLAY={"imagenet":"ImageNet","full_human_mae":"Full Human MAE","constrained_human_mae_last2":"Constrained Human MAE"}

def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--python",default=sys.executable);p.add_argument("--data-root",type=Path,default=Path("data/cat_dataset"));p.add_argument("--runs-dir",type=Path,default=Path("runs/final_cat_segmentation_5fold"));p.add_argument("--results-dir",type=Path,default=Path("results/final_cat_segmentation_5fold"));p.add_argument("--num-workers",type=int,default=0,help="Default 0 avoids Windows worker permission failures; data and optimization are unchanged.");p.add_argument("--dry-run",action="store_true");p.add_argument("--extended-seeds",action="store_true");p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);return p.parse_args()
def rows(p):return list(csv.DictReader(Path(p).open(encoding="utf-8")))
def read_json(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p,data,fields=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=fields or list(data[0]));w.writeheader();w.writerows(data)
def full_root(seed):return Path("runs/human_mae_recipe_ablation/baseline") if seed==0 else Path(f"runs/human_mae_adaptation_depth/full/seed{seed}/mae")
def last2_root(seed):return Path(f"runs/human_mae_adaptation_depth/last2/seed{seed}/mae")

def validate_human(root,seed,last2):
 root=Path(root);required=("config.json","metrics.csv","last_encoder.pt","best_encoder.pt","validation_metrics.json","validation_reconstruction.png");missing=[x for x in required if not (root/x).is_file()]
 if missing:raise RuntimeError(f"Missing Human SSL artifacts {root}: {missing}")
 c=read_json(root/"config.json");r=rows(root/"metrics.csv");p=torch.load(root/"last_encoder.pt",map_location="cpu",weights_only=False);errors=[];expected_depth=2 if last2 else None
 if int(c.get("seed",-1))!=seed:errors.append("config seed")
 if c.get("encoder_trainable_last_blocks")!=expected_depth:errors.append("adaptation depth")
 if int(r[-1]["epoch"])!=int(c["epochs"])-1:errors.append("metrics incomplete")
 if int(p.get("epoch",-1))!=int(c["epochs"])-1:errors.append("checkpoint incomplete")
 if p.get("adaptation")!="human_kidney_ultrasound_mae":errors.append("adaptation metadata")
 pc=p.get("config",{});keys=("human1_root","human2_root","human3_root","val_fraction","mask_ratio","norm_pixel_loss")
 for k in keys:
  checkpoint_value=pc.get(k,c.get(k));config_value=c.get(k)
  if k.endswith("_root"):
   if Path(checkpoint_value).resolve()!=Path(config_value).resolve():errors.append(f"checkpoint config {k}")
  elif checkpoint_value!=config_value:errors.append(f"checkpoint config {k}")
 if last2 and c.get("trainable_blocks")!=[10,11]:errors.append("trainable block list")
 if errors:raise RuntimeError(f"Invalid Human checkpoint {root}: {', '.join(errors)}")
 return (root/"last_encoder.pt").resolve(),c,p

def verify_shapes(full,last2):
 a=torch.load(full,map_location="cpu",weights_only=False)["state_dict"];b=torch.load(last2,map_location="cpu",weights_only=False)["state_dict"]
 missing=sorted(set(a)-set(b));unexpected=sorted(set(b)-set(a));shape=sorted(k for k in set(a)&set(b) if a[k].shape!=b[k].shape)
 if missing or unexpected or shape:raise RuntimeError(f"Last2 encoder mismatch: missing={len(missing)} unexpected={len(unexpected)} shape={len(shape)}")

def expected(method,fold,seed,checkpoint,a):
 return {"task":"segmentation","encoder":"vit_b16","encoder_init":"imagenet" if method=="imagenet" else "human_mae","encoder_checkpoint":None if checkpoint is None else str(checkpoint),"transfer":"full","data_root":str(a.data_root),"num_folds":5,"fold":fold,"split_seed":42,"seed":seed,"batch_size":8,"epochs":50,"lr":1e-4,"weight_decay":1e-4,"amp":a.amp}
def same_path(a,b):
 if a is None or b is None:return a is b
 return Path(a).resolve()==Path(b).resolve()
def complete(root,e,subjects):
 root=Path(root);needed=("config.json","metrics.csv","last.pt","best.pt","validation_metrics.json","validation_segmentation_preview.png","train_subjects.txt","val_subjects.txt","parameter_counts.json","trainable_parameters.json")
 if not all((root/x).is_file() for x in needed):return False
 try:c=read_json(root/"config.json");r=rows(root/"metrics.csv");p=torch.load(root/"last.pt",map_location="cpu",weights_only=False)
 except Exception:return False
 for k,v in e.items():
  if k=="encoder_checkpoint":
   if not same_path(c.get(k),v):return False
  elif c.get(k)!=v:return False
 if not r or len(r)!=e["epochs"] or int(r[-1]["epoch"])!=e["epochs"]-1 or int(p.get("epoch",-1))!=e["epochs"]-1:return False
 if c.get("classes")!=["background","cortex","medulla"]:return False
 if tuple(Path(x).name for x in c.get("train_subjects",[]))!=subjects[0] or tuple(Path(x).name for x in c.get("val_subjects",[]))!=subjects[1]:return False
 if (root/"train_subjects.txt").read_text(encoding="utf-8").splitlines()!=list(subjects[0]) or (root/"val_subjects.txt").read_text(encoding="utf-8").splitlines()!=list(subjects[1]):return False
 if e["encoder_init"]=="human_mae":
  s=c.get("encoder_load_summary",{})
  if s.get("missing_keys") or s.get("unexpected_keys") or s.get("shape_mismatch_keys") or not same_path(s.get("checkpoint"),e["encoder_checkpoint"]):return False
 return True

def candidates(method,fold,seed,a):
 target=a.runs_dir/method/f"fold{fold}"/f"seed{seed}";known=[]
 if fold==0:
  if method=="imagenet":known.append(Path(f"runs/cat_cross_species_anchor_validation/imagenet/segmentation/vit_b16/full/fold_0/seed_{seed}"))
  elif method=="full_human_mae":known.append(Path(f"runs/cat_cross_species_anchor_validation/human_mae_full/segmentation/vit_b16/full/fold_0/seed_{seed}/init_human_mae"))
  else:known.append(Path(f"runs/constrained_human_mae_cat_transfer/last2/seed{seed}/segmentation/vit_b16/full/fold_0/seed_{seed}/init_human_mae"))
 return [target,*known]
def command(method,fold,seed,checkpoint,target,a):
 c=[a.python,"-m","src.segmentation.train","--encoder","vit_b16","--encoder-init","imagenet" if method=="imagenet" else "human_mae","--transfer","full","--data-root",str(a.data_root),"--num-folds","5","--fold",str(fold),"--split-seed","42","--seed",str(seed),"--batch-size","8","--epochs","50","--lr","1e-4","--weight-decay","1e-4","--num-workers",str(a.num_workers),"--exact-run-dir",str(target),"--amp" if a.amp else "--no-amp"]
 if checkpoint is not None:c.extend(["--encoder-checkpoint",str(checkpoint)])
 return c
def best(root):
 r=max(rows(Path(root)/"metrics.csv"),key=lambda x:float(x["validation_mean_foreground_dice"]));return r
def mean_sd(v):v=list(v);return statistics.mean(v),statistics.stdev(v)
def signed_rank(values):
 try:return float(wilcoxon(values,alternative="two-sided",method="exact").pvalue)
 except ValueError:return 1.0
def paired(index,left,right,name,results):
 out=[]
 for fold in range(5):
  a=index[(left,fold,0)];b=index[(right,fold,0)];out.append({"fold":fold,"imagenet_dice" if left=="imagenet" else "full_mae_dice":a["val_dice"],"last2_dice" if right=="constrained_human_mae_last2" else "full_mae_dice":b["val_dice"],"delta_dice":b["val_dice"]-a["val_dice"],"imagenet_iou" if left=="imagenet" else "full_mae_iou":a["val_iou"],"last2_iou" if right=="constrained_human_mae_last2" else "full_mae_iou":b["val_iou"],"delta_iou":b["val_iou"]-a["val_iou"]})
 write(results/f"{name}.csv",out);d=[x["delta_dice"] for x in out];i=[x["delta_iou"] for x in out];summary={"mean_delta_dice":statistics.mean(d),"std_delta_dice":statistics.stdev(d),"median_delta_dice":statistics.median(d),"positive_fold_count":sum(x>0 for x in d),"negative_fold_count":sum(x<0 for x in d),"total_folds":5,"mean_delta_iou":statistics.mean(i),"std_delta_iou":statistics.stdev(i),"wilcoxon_dice_p_value":signed_rank(d)};write(results/f"{name}_summary.csv",[summary]);return out,summary

def aggregate(records,results):
 write(results/"all_fold_results.csv",records);index={(r["method"],r["fold"],r["seed"]):r for r in records};summary=[]
 for m in METHODS:
  x=[index[(m,f,0)] for f in range(5)];dm,ds=mean_sd(r["val_dice"] for r in x);im,is_=mean_sd(r["val_iou"] for r in x);lm,ls=mean_sd(r["val_loss"] for r in x);summary.append({"method":m,"n_folds":5,"mean_dice":dm,"std_dice":ds,"median_dice":statistics.median(r["val_dice"] for r in x),"mean_iou":im,"std_iou":is_,"mean_loss":lm,"std_loss":ls})
 write(results/"method_summary.csv",summary);lvi,lvis=paired(index,"imagenet","constrained_human_mae_last2","last2_vs_imagenet",results);lvf,lvfs=paired(index,"full_human_mae","constrained_human_mae_last2","last2_vs_full_mae",results)
 fvi=[]
 for f in range(5):
  a=index[("imagenet",f,0)];b=index[("full_human_mae",f,0)];fvi.append({"fold":f,"imagenet_dice":a["val_dice"],"full_mae_dice":b["val_dice"],"delta_dice":b["val_dice"]-a["val_dice"]})
 write(results/"full_mae_vs_imagenet.csv",fvi);d=[x["delta_dice"] for x in fvi];fvis={"mean_delta_dice":statistics.mean(d),"std_delta_dice":statistics.stdev(d),"median_delta_dice":statistics.median(d),"positive_fold_count":sum(x>0 for x in d),"negative_fold_count":sum(x<0 for x in d),"total_folds":5,"wilcoxon_dice_p_value":signed_rank(d)};write(results/"full_mae_vs_imagenet_summary.csv",[fvis])
 sidx={r["method"]:r for r in summary};publication=[]
 for m in METHODS:
  s=sidx[m];comp={"imagenet":{"mean_delta_dice":0,"positive_fold_count":""},"full_human_mae":fvis,"constrained_human_mae_last2":lvis}[m];publication.append({"method":DISPLAY[m],"dice_mean":s["mean_dice"],"dice_std":s["std_dice"],"iou_mean":s["mean_iou"],"iou_std":s["std_iou"],"mean_delta_vs_imagenet":comp["mean_delta_dice"],"positive_folds_vs_imagenet":comp["positive_fold_count"]})
 write(results/"publication_table.csv",publication);labels=[DISPLAY[m] for m in METHODS];colors=("#4C78A8","#E45756","#54A24B")
 fig,ax=plt.subplots(figsize=(7,5))
 for f in range(5):ax.plot(labels,[index[(m,f,0)]["val_dice"] for m in METHODS],color="0.75",marker="o",alpha=.8)
 ax.scatter(labels,[sidx[m]["mean_dice"] for m in METHODS],s=100,c=colors,marker="D",zorder=3,label="mean");ax.set_ylabel("Cat validation Dice");ax.legend();fig.tight_layout();fig.savefig(results/"final_5fold_dice.png",dpi=220);plt.close(fig)
 for data,name,ylabel in ((lvi,"paired_last2_vs_imagenet.png","Last2 Dice - ImageNet Dice"),(lvf,"paired_last2_vs_full_mae.png","Last2 Dice - Full Human MAE Dice")):
  fig,ax=plt.subplots();ax.scatter([f"fold{x['fold']}" for x in data],[x["delta_dice"] for x in data]);ax.axhline(0,color="black",linewidth=1);ax.set_ylabel(ylabel);fig.tight_layout();fig.savefig(results/name,dpi=220);plt.close(fig)
 fig,ax=plt.subplots();means=[sidx[m]["mean_dice"] for m in METHODS];stds=[sidx[m]["std_dice"] for m in METHODS];ax.errorbar(labels,means,yerr=stds,fmt="D",capsize=5,color="black")
 for j,m in enumerate(METHODS):ax.scatter([j]*5,[index[(m,f,0)]["val_dice"] for f in range(5)],color=colors[j],alpha=.8)
 ax.set_ylabel("Cat validation Dice");fig.tight_layout();fig.savefig(results/"method_mean_sd.png",dpi=220);plt.close(fig)
 print("\nMethod                    Dice mean ± SD     Δ vs ImageNet\n----------------------------------------------------------")
 for m in METHODS:print(f"{DISPLAY[m]:<26}{sidx[m]['mean_dice']:.6f} ± {sidx[m]['std_dice']:.6f}   {publication[METHODS.index(m)]['mean_delta_vs_imagenet']:+.6f}")

def main():
 a=parse();methods=("imagenet","constrained_human_mae_last2") if a.extended_seeds else METHODS;seeds=(0,1,2) if a.extended_seeds else (0,);folds=range(5);a.results_dir.mkdir(parents=True,exist_ok=True);validated={};fail=[]
 for seed in seeds:
  try:
   full,fc,fp=validate_human(full_root(seed),seed,False);last,lc,lp=validate_human(last2_root(seed),seed,True);verify_shapes(full,last);validated[("full_human_mae",seed)]=full;validated[("constrained_human_mae_last2",seed)]=last
   for k in ("human1_root","human2_root","human3_root","val_fraction"):
    if fc.get(k)!=lc.get(k):raise RuntimeError(f"Human SSL split mismatch seed{seed}: {k}")
  except Exception as e:fail.append(str(e))
 if fail:raise RuntimeError("Human checkpoint preflight failed before Cat training:\n"+"\n".join(fail))
 print("Human checkpoints:")
 for seed in seeds:
  print(f"[reuse] full_human_mae / seed{seed} -> {validated[('full_human_mae',seed)]} (epoch 49)");print(f"[reuse] constrained_human_mae_last2 / seed{seed} -> {validated[('constrained_human_mae_last2',seed)]} (blocks [10, 11], epoch 49)")
 splits={}
 for fold in folds:
  tr,va,_,_=split_subjects(str(a.data_root),"four_class",5,fold,42);ts={x.subject_id for x in tr};vs={x.subject_id for x in va};overlap=ts&vs
  if overlap:raise RuntimeError(f"fold{fold} train/validation overlap: {overlap}")
  splits[fold]=(tuple(x.subject_id for x in tr),tuple(x.subject_id for x in va));print(f"Cat fold{fold}: train={len(tr)} validation={len(va)} intersection=0")
 plans=[]
 for seed in seeds:
  for method in methods:
   checkpoint=validated.get((method,seed));
   for fold in folds:
    e=expected(method,fold,seed,checkpoint,a);found=next((p for p in candidates(method,fold,seed,a) if complete(p,e,splits[fold])),None);target=a.runs_dir/method/f"fold{fold}"/f"seed{seed}";plans.append((method,fold,seed,checkpoint,e,found,target));print(f"[reuse] {method}/fold{fold}/seed{seed} -> {found}" if found else f"[missing -> run] {method}/fold{fold}/seed{seed} -> {target}")
 if a.dry_run:return
 records=[];reused=new=0
 for method,fold,seed,checkpoint,e,found,target in plans:
  root=found
  if root is None:
   print(f"[run] {method}/fold{fold}/seed{seed}");subprocess.run(command(method,fold,seed,checkpoint,target,a),cwd=ROOT,check=True);root=target;new+=1
   if not complete(root,e,splits[fold]):raise RuntimeError(f"Run finished but integrity check failed: {root}")
  else:reused+=1
  r=best(root);records.append({"method":method,"fold":fold,"seed":seed,"encoder_checkpoint":"imagenet_pretrained" if checkpoint is None else str(checkpoint),"best_epoch":int(r["epoch"]),"val_dice":float(r["validation_mean_foreground_dice"]),"val_iou":float(r["validation_mean_foreground_iou"]),"val_loss":float(r["validation_loss"]),"val_cortex_dice":float(r["validation_cortex_dice"]),"val_medulla_dice":float(r["validation_medulla_dice"]),"val_background_dice":float(r["validation_background_dice"]),"reused_existing_run":found is not None,"run_dir":str(Path(root).resolve())})
 if not a.extended_seeds:aggregate(records,a.results_dir)
 else:write(a.results_dir/"extended_seed_all_fold_results.csv",records)
 print(f"Reused Cat runs: {reused}\nNew Cat runs: {new}")
if __name__=="__main__":main()
