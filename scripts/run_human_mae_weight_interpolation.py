"""Global interpolation between ImageNet and Full Human MAE ViT-B/16 encoders."""
from __future__ import annotations
import argparse,csv,json,math,subprocess,sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import run_cat_cross_species_anchor_validation as catrun
from scripts import run_human_mae_cat_aware_anchor as aware
from src.human_ssl.mae import VisionMAE

ALPHAS=(0.0,0.1,0.25,0.5,0.75,1.0)
def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=("feasibility","reproduce-0p1"),default="feasibility");p.add_argument("--python",default=sys.executable);p.add_argument("--runs-dir",type=Path,default=Path("runs/human_mae_weight_interpolation"));p.add_argument("--results-dir",type=Path,default=Path("results/human_mae_weight_interpolation"));p.add_argument("--repro-runs-dir",type=Path,default=Path("runs/human_mae_alpha_0p1_reproducibility"));p.add_argument("--repro-results-dir",type=Path,default=Path("results/human_mae_alpha_0p1_reproducibility"));p.add_argument("--cat-data-root",type=Path,default=Path("data/cat_dataset"));p.add_argument("--num-workers",type=int,default=4);p.add_argument("--force",action="store_true");p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);return p.parse_args()
def label(alpha):return str(alpha).replace(".","p")
def readrows(p):return list(csv.DictReader(Path(p).open(encoding="utf-8")))
def write(p,data):
 with Path(p).open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def execute(cmd,name):print(f"\n{'='*72}\n{name}\n{'='*72}\n{subprocess.list2cmdline(cmd)}",flush=True);subprocess.run(cmd,cwd=ROOT,check=True)
def best_cat(root):
 r=max(readrows(root/"metrics.csv"),key=lambda x:float(x["validation_mean_foreground_dice"]));return float(r["validation_mean_foreground_dice"]),float(r["validation_mean_foreground_iou"]),float(r["validation_loss"])
def drift(path):return sum(float(r["drift_1_minus_cka"]) for r in readrows(path))/4
def interpolation_cat_root(base,alpha):
 method=f"alpha_{label(alpha)}";suffix=f"init_human_mae_interp_a{round(alpha*100):03d}"
 return base/"cat_downstream"/method/"segmentation"/"vit_b16"/"full"/"fold_0"/"seed_0"/suffix

def generate_alpha_0p1(a,seed):
 human_checkpoint=Path(catrun.validate_human_ssl(catrun.full_ssl(seed),seed,False)).resolve()
 if seed==0:
  path=(a.runs_dir/"alpha_0p1"/"encoder.pt").resolve();payload=torch.load(path,map_location="cpu",weights_only=False)
  if not (payload.get("interpolation") and float(payload.get("interpolation_alpha",-1))==.1 and Path(payload.get("interpolation_ssl_source","")).resolve()==human_checkpoint):raise RuntimeError("Existing alpha=0.1 seed0 checkpoint source mismatch")
 else:path=(a.repro_runs_dir/f"seed{seed}"/"encoder.pt").resolve()
 model=VisionMAE("vit_b16",256,4,8,False);base={k:v.detach().cpu().float() for k,v in model.encoder.state_dict().items()};human_payload=torch.load(human_checkpoint,map_location="cpu",weights_only=False);human=human_payload["state_dict"]
 missing=sorted(set(base)-set(human));unexpected=sorted(set(human)-set(base));shape=sorted(k for k in set(base)&set(human) if base[k].shape!=human[k].shape)
 print(f"seed {seed}: matched={len(base)-len(shape)} missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape)}")
 if missing or unexpected or shape:raise RuntimeError(f"Encoder mismatch for seed {seed}")
 expected={k:base[k]+.1*(human[k].detach().cpu().float()-base[k]) for k in base}
 if seed>0 or a.force:
  path.parent.mkdir(parents=True,exist_ok=True);torch.save({"format":"feline_transfer_learning.vision_encoder.v1","encoder_name":"vit_b16_imagenet","initialization":"ImageNet-1K supervised","adaptation":"human_kidney_ultrasound_mae","epoch":49,"state_dict":expected,"interpolation":True,"interpolation_base":"ImageNet ViT-B/16","interpolation_ssl_source":str(human_checkpoint),"interpolation_alpha":.1,"human_mae_seed":seed,"config":{"seed":seed,"interpolation_alpha":.1,"source_human_checkpoint":str(human_checkpoint)}},path)
 saved=torch.load(path,map_location="cpu",weights_only=False)["state_dict"];max_diff=max(float((saved[k].float()-expected[k]).abs().max()) for k in base);full_norm=math.sqrt(sum(float((human[k].float()-base[k]).double().square().sum()) for k in base));update_norm=math.sqrt(sum(float((saved[k].float()-base[k]).double().square().sum()) for k in base));relative=update_norm/full_norm
 if max_diff!=0:raise RuntimeError(f"Interpolation numerical mismatch seed {seed}: {max_diff}")
 print(f"seed {seed}: max_abs_diff={max_diff} relative_update_norm={relative}")
 return path,human_checkpoint,max_diff,relative

def repro_cat_root(base,seed):return base/"cat_downstream"/f"seed{seed}"/"segmentation"/"vit_b16"/"full"/"fold_0"/f"seed_{seed}"/"init_human_mae_interp_a010"

def reproduce_0p1(a):
 a.repro_runs_dir.mkdir(parents=True,exist_ok=True);a.repro_results_dir.mkdir(parents=True,exist_ok=True);generated={};human_sources={}
 for seed in (0,1,2):generated[seed],human_sources[seed],_diff,_relative=generate_alpha_0p1(a,seed)
 cat_roots={};reused={};new=[]
 for seed in (0,1,2):
  for method in ("imagenet","human_mae_alpha_0p1","full_human_mae"):
   if method=="imagenet":root=catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),"imagenet",seed);checkpoint=None;expected_method="imagenet"
   elif method=="full_human_mae":root=catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),"human_mae_full",seed);checkpoint=human_sources[seed];expected_method="human_mae_full"
   elif seed==0:root=interpolation_cat_root(a.runs_dir,.1);checkpoint=generated[seed];expected_method="alpha_0p1"
   else:root=repro_cat_root(a.repro_runs_dir,seed);checkpoint=generated[seed];expected_method="alpha_0p1"
   expected=catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),expected_method,seed,checkpoint)
   complete=catrun.cat_complete(root,expected)
   if method!="human_mae_alpha_0p1" and not complete:raise RuntimeError(f"Incomplete existing control {method} seed {seed}")
   if method=="human_mae_alpha_0p1" and (a.force or not complete):
    output=a.runs_dir/"cat_downstream"/"alpha_0p1" if seed==0 else a.repro_runs_dir/"cat_downstream"/f"seed{seed}"
    cmd=[a.python,"-m","src.segmentation.train","--encoder","vit_b16","--encoder-init","human_mae","--encoder-checkpoint",str(checkpoint),"--transfer","full","--data-root",str(a.cat_data_root),"--num-folds","5","--fold","0","--split-seed","42","--seed",str(seed),"--batch-size","8","--epochs","50","--lr","1e-4","--weight-decay","1e-4","--num-workers",str(a.num_workers),"--output-dir",str(output),"--amp" if a.amp else "--no-amp"];execute(cmd,f"alpha=0.1 Cat segmentation seed {seed}")
    if not catrun.cat_complete(root,expected):raise RuntimeError(f"Incomplete alpha=0.1 Cat run seed {seed}")
    new.append(seed);complete=False
   cat_roots[(method,seed)]=root;reused[(method,seed)]=complete
 records=[]
 for method in ("imagenet","human_mae_alpha_0p1","full_human_mae"):
  for seed in (0,1,2):
   dice,iou,loss=best_cat(cat_roots[(method,seed)]);records.append({"method":method,"seed":seed,"human_mae_seed":"" if method=="imagenet" else seed,"alpha":0 if method=="imagenet" else .1 if method=="human_mae_alpha_0p1" else 1,"encoder_checkpoint":"imagenet_pretrained" if method=="imagenet" else str(generated[seed] if method=="human_mae_alpha_0p1" else human_sources[seed]),"cat_val_dice":dice,"cat_val_iou":iou,"cat_val_loss":loss,"reused_cat_run":reused[(method,seed)]})
 write(a.repro_results_dir/"all_seed_results.csv",records);index={(r["method"],r["seed"]):r for r in records};summaries=[]
 for method in ("imagenet","human_mae_alpha_0p1","full_human_mae"):
  selected=[index[(method,s)] for s in (0,1,2)];dm,ds=catrun.mean_std(r["cat_val_dice"] for r in selected);im,is_=catrun.mean_std(r["cat_val_iou"] for r in selected);lm,ls=catrun.mean_std(r["cat_val_loss"] for r in selected);summaries.append({"method":method,"n_seeds":3,"mean_dice":dm,"std_dice":ds,"mean_iou":im,"std_iou":is_,"mean_loss":lm,"std_loss":ls})
 write(a.repro_results_dir/"method_summary.csv",summaries);comparisons={}
 for base,stem,prefix in (("imagenet","alpha_0p1_vs_imagenet","imagenet"),("full_human_mae","alpha_0p1_vs_full","full")):
  paired=[]
  for seed in (0,1,2):
   b=index[(base,seed)];x=index[("human_mae_alpha_0p1",seed)];paired.append({"seed":seed,f"{prefix}_dice":b["cat_val_dice"],"alpha_0p1_dice":x["cat_val_dice"],"delta_dice":x["cat_val_dice"]-b["cat_val_dice"]})
  write(a.repro_results_dir/f"{stem}.csv",paired);values=[r["delta_dice"] for r in paired];write(a.repro_results_dir/f"{stem}_summary.csv",[{"mean_delta_dice":sum(values)/3,"std_delta_dice":torch.tensor(values,dtype=torch.float64).std(unbiased=True).item(),"positive_delta_count":sum(x>0 for x in values),"negative_delta_count":sum(x<0 for x in values),"total_seeds":3}]);comparisons[base]=paired
 labels=("ImageNet","alpha=0.1","Full Human MAE");fig,ax=plt.subplots()
 for seed in (0,1,2):ax.plot(labels,[index[(m,seed)]["cat_val_dice"] for m in ("imagenet","human_mae_alpha_0p1","full_human_mae")],marker="o",label=f"seed {seed}")
 ax.set(ylabel="Cat validation Dice");ax.legend();fig.tight_layout();fig.savefig(a.repro_results_dir/"alpha_0p1_seed_reproducibility.png",dpi=200);plt.close(fig)
 fig,ax=plt.subplots()
 for base,p in comparisons.items():ax.plot([r["seed"] for r in p],[r["delta_dice"] for r in p],marker="o",label=f"alpha0.1 - {base}")
 ax.axhline(0,color="black",linestyle="--");ax.set(xlabel="Seed",ylabel="Paired Dice delta",xticks=[0,1,2]);ax.legend();fig.tight_layout();fig.savefig(a.repro_results_dir/"alpha_0p1_paired_delta.png",dpi=200);plt.close(fig)
 print("Method               Mean Dice       SD\n-----------------------------------------");[print(f"{r['method']:<20} {r['mean_dice']:.6f}  {r['std_dice']:.6f}") for r in summaries]
 for base,p in comparisons.items():print(f"alpha=0.1 - {base}: "+", ".join(f"seed{r['seed']}={r['delta_dice']:.6f}" for r in p)+f", mean={sum(r['delta_dice'] for r in p)/3:.6f}")
 print("New Cat runs: "+", ".join(map(str,new)));print("Reused Cat runs: "+", ".join(f"{m}/seed{s}" for (m,s),v in reused.items() if v))

def generate(a):
 human_root=catrun.full_ssl(0);human_checkpoint=catrun.validate_human_ssl(human_root,0,False);payload=torch.load(human_checkpoint,map_location="cpu",weights_only=False);human=payload["state_dict"]
 model=VisionMAE("vit_b16",256,4,8,False);pretrained={k:v.detach().cpu().float() for k,v in model.encoder.state_dict().items()}
 missing=sorted(set(pretrained)-set(human));unexpected=sorted(set(human)-set(pretrained));shape=[k for k in set(pretrained)&set(human) if pretrained[k].shape!=human[k].shape]
 print(f"matched encoder parameters: {len(pretrained)-len(shape)}\nmissing: {len(missing)}\nunexpected: {len(unexpected)}\nshape mismatch: {len(shape)}")
 if missing or unexpected or shape:raise RuntimeError("ImageNet/Human encoder state mismatch")
 delta={k:human[k].detach().cpu().float()-pretrained[k] for k in pretrained};full_norm=math.sqrt(sum(float(v.double().square().sum()) for v in delta.values()));info={}
 for alpha in ALPHAS:
  root=a.runs_dir/f"alpha_{label(alpha)}";root.mkdir(parents=True,exist_ok=True);path=root/"encoder.pt"
  if alpha==0.0:state={k:v.clone() for k,v in pretrained.items()}
  elif alpha==1.0:state={k:human[k].detach().cpu().float().clone() for k in pretrained}
  else:state={k:pretrained[k]+alpha*delta[k] for k in pretrained}
  checkpoint={"format":"feline_transfer_learning.vision_encoder.v1","encoder_name":"vit_b16_imagenet","initialization":"ImageNet-1K supervised","adaptation":"human_kidney_ultrasound_mae","epoch":49,"state_dict":state,"model_state_dict":{k.removeprefix('model.'):v for k,v in state.items() if k.startswith('model.')},"interpolation":True,"interpolation_base":"ImageNet ViT-B/16","interpolation_ssl_source":str(human_checkpoint),"interpolation_alpha":alpha,"config":{"seed":0,"interpolation_alpha":alpha,"source_human_checkpoint":str(human_checkpoint)}}
  torch.save(checkpoint,path);norm=math.sqrt(sum(float((state[k]-pretrained[k]).double().square().sum()) for k in state));info[alpha]=(path.resolve(),norm,norm/full_norm if full_norm else 0)
 max0=max(float((torch.load(info[0.0][0],map_location="cpu",weights_only=False)["state_dict"][k]-pretrained[k]).abs().max()) for k in pretrained);max1=max(float((torch.load(info[1.0][0],map_location="cpu",weights_only=False)["state_dict"][k]-human[k].float()).abs().max()) for k in pretrained)
 print(f"max_abs_diff(alpha0, ImageNet): {max0}\nmax_abs_diff(alpha1, HumanMAE): {max1}")
 if max0!=0 or max1!=0:raise RuntimeError("Interpolation endpoint identity check failed")
 return info,Path(human_checkpoint),max0,max1

def main():
 a=parse()
 if a.mode=="reproduce-0p1":
  reproduce_0p1(a)
  return
 a.runs_dir.mkdir(parents=True,exist_ok=True);a.results_dir.mkdir(parents=True,exist_ok=True);info,human_checkpoint,max0,max1=generate(a);source=json.loads(Path("checkpoints/human_mae_vit_b16_trajectory/config.json").read_text(encoding="utf-8"));cat_roots={};reused={};new=[]
 for alpha in ALPHAS:
  checkpoint=info[alpha][0]
  if alpha in (0.0,1.0):
   method="imagenet" if alpha==0 else "human_mae_full";root=catrun.run_dir(argparse.Namespace(runs_dir=Path("runs/cat_cross_species_anchor_validation"),fold=0),method,0);expected_checkpoint=None if alpha==0 else human_checkpoint;expected=catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),method,0,expected_checkpoint)
   if not catrun.cat_complete(root,expected):raise RuntimeError(f"Incomplete endpoint Cat control alpha={alpha}")
   reused[alpha]=True
  else:
   method=f"alpha_{label(alpha)}";root=interpolation_cat_root(a.runs_dir,alpha);expected=catrun.expected_config(argparse.Namespace(data_root=a.cat_data_root,fold=0,amp=a.amp),method,0,checkpoint)
   complete=catrun.cat_complete(root,expected)
   if a.force or not complete:
    cmd=[a.python,"-m","src.segmentation.train","--encoder","vit_b16","--encoder-init","human_mae","--encoder-checkpoint",str(checkpoint),"--transfer","full","--data-root",str(a.cat_data_root),"--num-folds","5","--fold","0","--split-seed","42","--seed","0","--batch-size","8","--epochs","50","--lr","1e-4","--weight-decay","1e-4","--num-workers",str(a.num_workers),"--output-dir",str(a.runs_dir/"cat_downstream"/method),"--amp" if a.amp else "--no-amp"];execute(cmd,f"Cat segmentation interpolation alpha={alpha}");new.append(alpha)
    if not catrun.cat_complete(root,expected):raise RuntimeError(f"Incomplete Cat run alpha={alpha}")
   reused[alpha]=complete and not a.force
  cat_roots[alpha]=root
 diagnostics=a.runs_dir/"diagnostics";diagnostics.mkdir(exist_ok=True);summary=[]
 for alpha in ALPHAS:
  human_path=diagnostics/f"alpha_{label(alpha)}_human.csv";cat_path=diagnostics/f"alpha_{label(alpha)}_cat_train.csv";aware.diagnostic(a,source,info[alpha][0],human_path,cat_path,0);dice,iou,loss=best_cat(cat_roots[alpha]);summary.append({"alpha":alpha,"encoder_checkpoint":str(info[alpha][0]),"parameter_update_norm":info[alpha][1],"relative_update_norm":info[alpha][2],"mean_human_feature_drift":drift(human_path),"mean_cat_train_feature_drift":drift(cat_path),"cat_val_dice":dice,"cat_val_iou":iou,"cat_val_loss":loss,"reused_cat_run":reused[alpha]})
 write(a.results_dir/"interpolation_summary.csv",summary);image=summary[0]["cat_val_dice"];full=summary[-1]["cat_val_dice"];write(a.results_dir/"interpolation_deltas.csv",[{"alpha":r["alpha"],"dice_minus_imagenet":r["cat_val_dice"]-image,"dice_minus_full_human_mae":r["cat_val_dice"]-full} for r in summary])
 for xkey,ykey,name,xlabel,ylabel in (("alpha","cat_val_dice","alpha_vs_cat_dice.png","alpha","Cat validation Dice"),("alpha","mean_cat_train_feature_drift","alpha_vs_cat_drift.png","alpha","Mean Cat-train feature drift"),("alpha","mean_human_feature_drift","alpha_vs_human_drift.png","alpha","Mean Human feature drift"),("relative_update_norm","cat_val_dice","update_norm_vs_cat_dice.png","Relative update norm","Cat validation Dice")):
  fig,ax=plt.subplots();ax.plot([r[xkey] for r in summary],[r[ykey] for r in summary],marker="o");
  if name=="alpha_vs_cat_dice.png":ax.axhline(image,color="black",linestyle="--")
  ax.set(xlabel=xlabel,ylabel=ylabel);fig.tight_layout();fig.savefig(a.results_dir/name,dpi=200);plt.close(fig)
 print("Alpha      Cat Dice      Delta vs ImageNet\n--------------------------------------");[print(f"{r['alpha']:<10.2f} {r['cat_val_dice']:.6f}  {r['cat_val_dice']-image:+.6f}") for r in summary];intermediate=max(summary[1:-1],key=lambda r:r["cat_val_dice"]);print(f"Best intermediate alpha: {intermediate['alpha']}");print("New Cat runs: "+", ".join(map(str,new)));print("Reused Cat runs: "+", ".join(str(k) for k,v in reused.items() if v));print(f"Endpoint sanity: alpha0={max0}, alpha1={max1}")
if __name__=="__main__":main()
