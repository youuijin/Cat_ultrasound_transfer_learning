"""Full Human ViT-B/16 Barlow Twins adaptation."""
from __future__ import annotations
import argparse,csv,json,math
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from src.classification.training_utils import set_seed
from src.human_ssl.data import DEFAULT_ROOTS,build_barlow_loaders,dataset_counts,discover_ssl_samples,split_ssl_samples
from src.human_ssl.mae import VisionMAE
from src.human_ssl.feature_anchor import encoder_checksum,feature_preservation_loss

def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--human1-root",type=Path,default=DEFAULT_ROOTS["human1"]);p.add_argument("--human2-root",type=Path,default=DEFAULT_ROOTS["human2"]);p.add_argument("--human3-root",type=Path,default=DEFAULT_ROOTS["human3"]);p.add_argument("--val-fraction",type=float,default=.1);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--epochs",type=int,default=50);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--weight-decay",type=float,default=.05);p.add_argument("--warmup-epochs",type=int,default=10);p.add_argument("--num-workers",type=int,default=4);p.add_argument("--seed",type=int,default=0);p.add_argument("--barlow-projector-dim",type=int,default=2048);p.add_argument("--barlow-lambda-offdiag",type=float,default=.005);p.add_argument("--feature-anchor-lambda",type=float,default=0.0);p.add_argument("--output-dir",type=Path,default=Path("runs/human_barlow/full/seed0"));p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True);a=p.parse_args()
 if a.feature_anchor_lambda<0:p.error("--feature-anchor-lambda must be non-negative")
 return a
class Projector(nn.Sequential):
 def __init__(self,input_dim,dim):super().__init__(nn.Linear(input_dim,dim,bias=False),nn.BatchNorm1d(dim),nn.ReLU(inplace=True),nn.Linear(dim,dim,bias=False),nn.BatchNorm1d(dim),nn.ReLU(inplace=True),nn.Linear(dim,dim,bias=False))
def off_diagonal(x):
 n,m=x.shape
 if n!=m:raise ValueError("Cross-correlation matrix must be square")
 return x.flatten()[:-1].view(n-1,n+1)[:,1:].flatten()
def barlow_twins_loss(a,b,weight):
 an=(a-a.mean(0))/(a.std(0,unbiased=False)+1e-6);bn=(b-b.mean(0))/(b.std(0,unbiased=False)+1e-6);c=an.T@bn/a.shape[0];on=torch.diagonal(c).add(-1).square().sum();off=off_diagonal(c).square().sum();return on+weight*off,on,off,c
def epoch(encoder,projector,loader,device,scaler,amp,weight,teacher=None,anchor_weight=0.0,optimizer=None):
 training=optimizer is not None;encoder.train(training);projector.train(training)
 if teacher is not None:teacher.eval()
 tot={k:0. for k in ("loss","barlow_loss","feature_anchor_loss","on_diag_loss","off_diag_loss","projector_feature_std_mean","projector_feature_std_min","mean_abs_offdiag_correlation","mean_diagonal_correlation")};count=0
 with torch.enable_grad() if training else torch.no_grad():
  for view1,view2,_ in loader:
   view1=view1.to(device,non_blocking=True);view2=view2.to(device,non_blocking=True)
   if training:optimizer.zero_grad(set_to_none=True)
   with torch.amp.autocast(device_type=device.type,enabled=amp):
    mean=torch.tensor(encoder.preprocess.mean,device=device)[None,:,None,None];std=torch.tensor(encoder.preprocess.std,device=device)[None,:,None,None];x1=(view1-mean)/std;x2=(view2-mean)/std
    z1=projector(encoder.forward_features(x1));z2=projector(encoder.forward_features(x2));barlow,on,off,c=barlow_twins_loss(z1,z2,weight)
    anchor=barlow.new_zeros(())
    if teacher is not None:
     anchor1,_=feature_preservation_loss(encoder,teacher,x1,tuple(range(12)));anchor2,_=feature_preservation_loss(encoder,teacher,x2,tuple(range(12)));anchor=(anchor1+anchor2)/2
    loss=barlow+anchor_weight*anchor
   if training:scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
   n=view1.shape[0];feature_std=torch.cat((z1,z2)).float().std(0,unbiased=False);vals={"loss":loss,"barlow_loss":barlow,"feature_anchor_loss":anchor,"on_diag_loss":on,"off_diag_loss":off,"projector_feature_std_mean":feature_std.mean(),"projector_feature_std_min":feature_std.min(),"mean_abs_offdiag_correlation":off_diagonal(c).abs().mean(),"mean_diagonal_correlation":torch.diagonal(c).mean()}
   for k,v in vals.items():tot[k]+=float(v)*n
   count+=n
 return {k:v/max(count,1) for k,v in tot.items()}
def save(path,encoder,args,epoch,val):torch.save({"format":"feline_transfer_learning.vision_encoder.v1","encoder_name":"vit_b16_imagenet","initialization":"ImageNet-1K supervised","adaptation":"human_kidney_ultrasound_barlow","epoch":epoch,"validation_barlow_loss":val,"state_dict":encoder.state_dict(),"config":vars(args)},path)
def main():
 a=parse();set_seed(a.seed);a.output_dir.mkdir(parents=True,exist_ok=True);samples=discover_ssl_samples({n:getattr(a,f"{n}_root") for n in ("human1","human2","human3")});train,val=split_ssl_samples(samples,a.val_fraction,a.seed);mae=VisionMAE("vit_b16",256,4,8,False);encoder=mae.encoder;teacher=deepcopy(encoder) if a.feature_anchor_lambda>0 else None
 if teacher is not None:
  teacher.requires_grad_(False);teacher.eval();teacher_checksum=encoder_checksum(teacher)
 projector=Projector(encoder.feature_dim,a.barlow_projector_dim);train_loader,val_loader=build_barlow_loaders(train,val,encoder.preprocess.image_size,a.batch_size,a.num_workers,a.seed)
 config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};config.update(ssl_method="barlow",encoder="vit_b16",transfer="full",barlow_feature_source="cls_token",feature_anchor_teacher="imagenet_pretrained" if teacher is not None else None,feature_anchor_layers=list(range(12)) if teacher is not None else [],feature_anchor_representation="mean_patch_tokens" if teacher is not None else None,augmentation={"random_resized_crop_scale":[.90,1.0],"ratio":[.95,1.05],"horizontal_flip_probability":.5,"rotation_degrees":5,"brightness":.12,"contrast":.12,"gamma":[.92,1.08],"gaussian_noise_probability":.3,"gaussian_noise_std":.01},train_images=len(train),validation_images=len(val),dataset_counts=dataset_counts(samples),trainable_encoder_params=sum(p.numel() for p in encoder.parameters() if p.requires_grad),total_encoder_params=sum(p.numel() for p in encoder.parameters()),projector_params=sum(p.numel() for p in projector.parameters()))
 (a.output_dir/"config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding="utf-8");first=next(iter(train_loader));print(f"encoder type: ViT-B/16\nencoder parameter count: {config['total_encoder_params']}\ntrainable encoder parameter count: {config['trainable_encoder_params']}\nprojector parameter count: {config['projector_params']}\nbarlow_feature_source = cls_token\nprojector dimension: {a.barlow_projector_dim}\nlambda_offdiag: {a.barlow_lambda_offdiag}\nbatch size: {a.batch_size}\nLR: {a.lr}\nepochs: {a.epochs}\nmean absolute pixel difference view1/view2: {float((first[0]-first[1]).abs().mean())}")
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");encoder.to(device);projector.to(device)
 if teacher is not None:teacher.to(device)
 optimizer=torch.optim.AdamW([*encoder.parameters(),*projector.parameters()],lr=a.lr,weight_decay=a.weight_decay)
 def factor(e):
  if e<a.warmup_epochs:return (e+1)/a.warmup_epochs
  return .5*(1+math.cos(math.pi*min(1.,(e-a.warmup_epochs)/max(1,a.epochs-a.warmup_epochs-1))))
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor);amp=a.amp and device.type=="cuda";scaler=torch.amp.GradScaler(device.type,enabled=amp);writer=SummaryWriter(a.output_dir/"tensorboard");history=[];best=float("inf")
 try:
  for i in range(a.epochs):
   tr=epoch(encoder,projector,train_loader,device,scaler,amp,a.barlow_lambda_offdiag,teacher,a.feature_anchor_lambda,optimizer);va=epoch(encoder,projector,val_loader,device,scaler,amp,a.barlow_lambda_offdiag,teacher,a.feature_anchor_lambda);lr=optimizer.param_groups[0]["lr"];scheduler.step();save(a.output_dir/"last_encoder.pt",encoder,a,i,va["loss"])
   if va["loss"]<best:best=va["loss"];save(a.output_dir/"best_encoder.pt",encoder,a,i,va["loss"])
   history.append({"epoch":i,**{f"train_{k}":v for k,v in tr.items()},**{f"validation_{k}":v for k,v in va.items()},"learning_rate":lr})
   with (a.output_dir/"metrics.csv").open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
   for phase,data in (("train",tr),("validation",va)):
    for k,v in data.items():writer.add_scalar(f"{phase}/{k}",v,i)
   writer.flush();print(f"epoch {i+1}/{a.epochs} train_barlow={tr['loss']:.6f} val_barlow={va['loss']:.6f}")
 finally:writer.close()
 if teacher is not None and encoder_checksum(teacher)!=teacher_checksum:raise RuntimeError("Frozen ImageNet anchor teacher changed during training")
if __name__=="__main__":main()
