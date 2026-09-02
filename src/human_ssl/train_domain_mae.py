"""Cat-only or balanced Human+Cat MAE with fixed all-block ImageNet anchoring."""
from __future__ import annotations

import argparse, csv, json, math
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from src.classification.training_utils import set_seed
from src.human_ssl.cat_anchor import build_cat_anchor_loader, write_subject_csv
from src.human_ssl.data import build_ssl_loaders, dataset_counts, discover_ssl_samples, split_ssl_samples
from src.human_ssl.feature_anchor import encoder_checksum, feature_preservation_loss, final_feature_drift, normalized_images
from src.human_ssl.mae import VisionMAE

BLOCKS=tuple(range(12)); COMMON_BLOCKS=(3,6,9,11)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--ssl-domains",choices=("cat_only","human_cat"),required=True)
    p.add_argument("--human1-root",type=Path,required=True); p.add_argument("--human2-root",type=Path,required=True); p.add_argument("--human3-root",type=Path,required=True)
    p.add_argument("--cat-data-root",type=Path,default=Path("data/cat_dataset")); p.add_argument("--val-fraction",type=float,default=.1)
    p.add_argument("--mask-ratio",type=float,default=.75); p.add_argument("--norm-pixel-loss",action=argparse.BooleanOptionalAction,default=False)
    p.add_argument("--decoder-dim",type=int,default=256); p.add_argument("--decoder-depth",type=int,default=4); p.add_argument("--decoder-heads",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=32); p.add_argument("--epochs",type=int,default=50); p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--weight-decay",type=float,default=.05); p.add_argument("--warmup-epochs",type=int,default=10); p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--seed",type=int,default=0); p.add_argument("--feature-anchor-lambda",type=float,default=.01); p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True); return p.parse_args()


def cycle_next(iterator,loader):
    try: return next(iterator),iterator
    except StopIteration:
        iterator=iter(loader); return next(iterator),iterator


def domain_loss(model,teacher,images,mask_ratio):
    mae,_,_=model(images,mask_ratio); anchor,_=feature_preservation_loss(model.encoder,teacher,normalized_images(model,images),BLOCKS)
    return mae,anchor,mae+.01*anchor


def train_epoch(model,teacher,human_loader,cat_loader,steps,device,scaler,amp,optimizer,mode,mask_ratio):
    model.train(); teacher.eval(); human_it=iter(human_loader); cat_it=iter(cat_loader); totals={k:0.0 for k in ("human_mae","human_anchor","cat_mae","cat_anchor","total")}
    for _ in range(steps):
        (cat,_),cat_it=cycle_next(cat_it,cat_loader); cat=cat.to(device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type,enabled=amp):
            cmae,canchor,closs=domain_loss(model,teacher,cat,mask_ratio)
            if mode=="human_cat":
                (human,_),human_it=cycle_next(human_it,human_loader); human=human.to(device,non_blocking=True); hmae,hanchor,hloss=domain_loss(model,teacher,human,mask_ratio); loss=.5*hloss+.5*closs
            else: hmae=hanchor=closs.new_zeros(()); loss=closs
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        for key,value in (("human_mae",hmae),("human_anchor",hanchor),("cat_mae",cmae),("cat_anchor",canchor),("total",loss)): totals[key]+=float(value)
    return {k:v/steps for k,v in totals.items()}


@torch.no_grad()
def evaluate(model,teacher,loader,device,amp,mask_ratio):
    model.eval(); teacher.eval(); mae=anch=count=0.0
    for images,_ in loader:
        images=images.to(device,non_blocking=True)
        with torch.amp.autocast(device_type=device.type,enabled=amp): m,a,_=domain_loss(model,teacher,images,mask_ratio)
        n=images.shape[0]; mae+=float(m)*n; anch+=float(a)*n; count+=n
    return mae/count,anch/count


def save(path,model,args,epoch,val_loss):
    torch.save({"format":"feline_transfer_learning.vision_encoder.v1","encoder_name":"vit_b16_imagenet","initialization":model.encoder.pretraining,
                "adaptation":"human_kidney_ultrasound_mae","ssl_domains":args.ssl_domains,"epoch":epoch,
                "validation_reconstruction_loss":val_loss,"state_dict":model.encoder.state_dict(),"model_state_dict":model.encoder.model.state_dict(),"config":vars(args)},path)


def main():
    a=parse_args()
    if a.feature_anchor_lambda != .01: raise ValueError("This feasibility experiment fixes feature anchor lambda at 0.01")
    set_seed(a.seed); a.output_dir.mkdir(parents=True,exist_ok=True)
    samples=discover_ssl_samples({n:getattr(a,f"{n}_root") for n in ("human1","human2","human3")}); human_train,human_val=split_ssl_samples(samples,a.val_fraction,a.seed)
    model=VisionMAE("vit_b16",a.decoder_dim,a.decoder_depth,a.decoder_heads,a.norm_pixel_loss); teacher=deepcopy(model.encoder).requires_grad_(False).eval(); initial_checksum=encoder_checksum(teacher)
    initial_difference=max(float((x.detach()-y.detach()).abs().max()) for x,y in zip(model.encoder.parameters(),teacher.parameters()))
    if initial_difference!=0: raise RuntimeError(f"Student/teacher ImageNet initialization mismatch: {initial_difference}")
    human_loader,human_val_loader=build_ssl_loaders(human_train,human_val,model.image_size,a.batch_size,a.num_workers,a.seed)
    cat_loader,cat_train,cat_val=build_cat_anchor_loader(a.cat_data_root,model.image_size,a.batch_size,a.num_workers,a.seed,5,0,42,True)
    cat_eval_loader,_t,_v=build_cat_anchor_loader(a.cat_data_root,model.image_size,a.batch_size,a.num_workers,a.seed,5,0,42,False)
    train_ids={x.subject_id for x in cat_train}; val_ids={x.subject_id for x in cat_val}; assert not train_ids&val_ids
    write_subject_csv(a.output_dir/"cat_ssl_train_subjects.csv",cat_train); write_subject_csv(a.output_dir/"cat_downstream_val_subjects.csv",cat_val)
    steps=len(human_loader); config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}; config.update(feature_anchor_layers=list(BLOCKS),human_cat_ssl_weight=.5 if a.ssl_domains=="human_cat" else 0,cat_ssl_weight=.5 if a.ssl_domains=="human_cat" else 1.0,optimizer_steps_per_epoch=steps,total_optimizer_steps=steps*a.epochs,cat_ssl_train_subject_count=len(train_ids),cat_downstream_val_subject_count=len(val_ids),cat_subject_overlap_count=0,cat_labels_used=False,cat_masks_used=False,human_train_images=len(human_train),human_validation_images=len(human_val),cat_train_images=len(cat_loader.dataset),shared_spatial_preprocessing="square zero-pad; 224x224 bicubic resize; random horizontal flip during training",input_source_difference="Human inputs are existing PNG intensities; Cat NIfTI inputs use existing image-only 1st/99th percentile uint8 conversion; no masks or labels")
    (a.output_dir/"config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Cat SSL train subject count: {len(train_ids)}\nCat downstream validation subject count: {len(val_ids)}\noverlap count: 0")
    print(f"Human batches per optimizer step: {1 if a.ssl_domains=='human_cat' else 0}\nCat batches per optimizer step: 1\nHuman loss weight: {config['human_cat_ssl_weight']}\nCat loss weight: {config['cat_ssl_weight']}\ntotal optimizer steps: {config['total_optimizer_steps']}")
    print(f"anchor layers: {list(BLOCKS)}\nteacher frozen: True\nstudent trainable encoder params: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)}\ninitial student/teacher max difference: {initial_difference}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device); teacher.to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    def factor(epoch):
        if epoch<a.warmup_epochs:return (epoch+1)/a.warmup_epochs
        progress=(epoch-a.warmup_epochs)/max(1,a.epochs-a.warmup_epochs-1); return .5*(1+math.cos(math.pi*min(1.,progress)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor); amp=a.amp and device.type=="cuda"; scaler=torch.amp.GradScaler(device.type,enabled=amp); writer=SummaryWriter(a.output_dir/"tensorboard"); history=[]; best=float("inf")
    try:
        for epoch in range(a.epochs):
            train=train_epoch(model,teacher,human_loader,cat_loader,steps,device,scaler,amp,optimizer,a.ssl_domains,a.mask_ratio)
            cmae,canchor=evaluate(model,teacher,cat_eval_loader,device,amp,a.mask_ratio)
            if a.ssl_domains=="human_cat": hmae,hanchor=evaluate(model,teacher,human_val_loader,device,amp,a.mask_ratio); selection=.5*(hmae+.01*hanchor)+.5*(cmae+.01*canchor)
            else: hmae=hanchor=float("nan"); selection=cmae+.01*canchor
            lr=optimizer.param_groups[0]["lr"]; scheduler.step(); save(a.output_dir/"last_encoder.pt",model,a,epoch,selection)
            if selection<best: best=selection; save(a.output_dir/"best_encoder.pt",model,a,epoch,selection)
            history.append({"epoch":epoch,**{f"train_{k}":v for k,v in train.items()},"validation_human_mae_loss":hmae,"validation_human_anchor_loss":hanchor,"validation_cat_mae_loss":cmae,"validation_cat_anchor_loss":canchor,"validation_selection_loss":selection,"learning_rate":lr,"optimizer_steps_completed":(epoch+1)*steps})
            with (a.output_dir/"ssl_metrics.csv").open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
            writer.add_scalar("validation/cat_mae_loss",cmae,epoch); writer.add_scalar("validation/human_mae_loss",hmae,epoch); writer.flush(); print(f"epoch {epoch+1}/{a.epochs} cat_mae={cmae:.6f} human_mae={hmae:.6f} total_steps={(epoch+1)*steps}")
        final_feature_drift(model,teacher,human_val_loader,device,a.output_dir/"human_feature_drift.csv",COMMON_BLOCKS); final_feature_drift(model,teacher,cat_eval_loader,device,a.output_dir/"cat_train_feature_drift.csv",COMMON_BLOCKS)
        if encoder_checksum(teacher)!=initial_checksum: raise RuntimeError("Frozen teacher checksum changed")
        print("Teacher checksum unchanged: verified")
    finally: writer.close()


if __name__=="__main__": main()
