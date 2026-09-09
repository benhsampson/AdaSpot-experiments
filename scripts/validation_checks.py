"""Inference, metrics, visuals, and a native AdaSpot training-step check."""
import copy
import gc
import hashlib
import json
import math
import random
import os
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.frame import ActionSpotVideoDataset, FrameReaderVideo
from util.eval import process_frame_predictions, non_maximum_supression, soft_non_maximum_supression
from util.score import compute_mAPs, compute_average_precision, get_predictions, parse_ground_truth
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'validation/results'
BASE=ROOT/'artifacts/finegym'
FRAMES=Path(os.environ.get('ADASPOT_FRAME_DIR','/tmp/adaspot-finegym-frames'))


def write(name,value):
    (RESULTS/name).write_text(json.dumps(value,indent=2)+'\n')


def seed():
    random.seed(1);np.random.seed(1);torch.manual_seed(1);torch.cuda.manual_seed_all(1)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=False


def gpu_memory():
    torch.cuda.synchronize()
    return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}


def data(rows,classes):
    p=BASE/'active_labels.json';p.write_text(json.dumps(rows))
    return ActionSpotVideoDataset(str(p),classes,str(FRAMES),100,dataset='finegym',overlap_len=50)


def inference(model,cfg,classes,stage):
    seed();model.clean_modules()
    rows=json.loads((ROOT/'validation/test_labels.json').read_text())
    if stage=='smoke':
        assert len(rows)>=2
        rows=rows[:2]
    else:
        assert len(rows)==20, 'Run download_subset.py to completion first'
    score_dir=ROOT/'artifacts/scores';score_dir.mkdir(exist_ok=True)
    checkpoint=json.loads((RESULTS/'FineGym_big_checkpoint.json').read_text())
    manifest=json.loads((ROOT/'validation/subset.json').read_text())
    inputs={x['original']['video']:x['sha256'] for x in manifest['clips'] if 'sha256' in x}
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['model','dataset','util'] for p in (ROOT/folder).rglob('*.py')}
    common={'checkpoint_sha256':checkpoint['sha256'],'config':cfg,'source_hashes':source_hashes,'pipeline_version':1}
    pred_dict={}; timings=[]
    for row in rows:
        vid=row['video']; cache=score_dir/f'{vid}.npz'
        dataset=data([row],classes)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        scores=np.zeros((row['num_frames'],len(classes)+1),np.float32)
        support=np.zeros(row['num_frames'],np.int32)
        signature=hashlib.sha256(json.dumps({**common,'input_video_sha256':inputs[vid],'labels':row},sort_keys=True).encode()).hexdigest()
        cache_used=cache.exists() and cache.with_suffix('.json').exists() and stage!='smoke'
        if cache_used:
            cache_used=json.loads(cache.with_suffix('.json').read_text()).get('signature')==signature
        if cache_used:
            old=np.load(cache);scores=old['scores'];support=old['support']
            prior=json.loads(cache.with_suffix('.json').read_text())
        else:
            windows=0
            for batch in DataLoader(dataset,batch_size=1,num_workers=2,pin_memory=True,prefetch_factor=1):
                _, predictions=model.predict(batch['frame'])
                assert predictions.shape==(1,100,len(classes)+1)
                assert np.isfinite(predictions).all()
                assert np.allclose(predictions.sum(-1),1,atol=0.01)
                start=int(batch['start'][0]);lo=max(start,0);hi=min(start+100,row['num_frames'])
                scores[lo:hi]+=predictions[0,lo-start:hi-start];support[lo:hi]+=1;windows+=1
            assert (support>0).all(),vid
            prior={'video':vid,'signature':signature,'windows':windows,'seconds':time.perf_counter()-started,**gpu_memory()}
            np.savez_compressed(cache,scores=scores,support=support)
            cache.with_suffix('.json').write_text(json.dumps(prior,indent=2))
        assert (support>0).all() and np.isfinite(scores).all()
        pred_dict[vid]=(scores.copy(),support.copy())
        timings.append({**prior,'cache_used':cache_used})
        print(json.dumps(timings[-1]),flush=True)
    dataset=data(rows,classes)
    err,f1,events,high_recall,averaged=process_frame_predictions(dataset,classes,pred_dict)
    variants={'raw':high_recall,'nms':non_maximum_supression(high_recall,1,0.01),
              'soft_nms':soft_non_maximum_supression(high_recall,2,0.01)}
    metrics={}
    truth=parse_ground_truth(rows)
    for name,pred in variants.items():
        maps,tols=compute_mAPs(rows,pred,printed=True)
        metrics[name]={'mAP':dict(zip(map(str,tols),map(float,maps))), 'mean_mAP':float(np.mean(maps)),
            'AP_per_class':{label:{str(t):compute_average_precision(get_predictions(pred,label),gt,t) for t in tols} for label,gt in truth.items()}}
    support_counts=Counter(e['label'] for r in rows for e in r['events'])
    metrics.update({'frame_accuracy':float(1-err.get()),'foreground_detection_F1_exact':float(f1.get(None)),
        'class_F1_exact':{k:float(f1.get(v)) for k,v in classes.items()},'class_support':dict(support_counts),
        'provenance':common,'background_only_frame_accuracy':1-sum(len(r['events']) for r in rows)/sum(r['num_frames'] for r in rows),'num_clips':len(rows),'num_frames':sum(r['num_frames'] for r in rows),'timings':timings,
        'precision':'BF16 autocast; FP32 overlap accumulation','batch_size':1,'clip_len':100,'overlap':50,
        'alignment_caveat':'Integer-second archive trim origin inferred from event IDs and video durations; full-source frame identity not available.',
        'seed':1})
    write(f'{stage}_metrics.json',metrics)
    write(f'{stage}_predictions.json',variants)
    write(f'{stage}_argmax_events.json',events)
    visuals(rows[:2],averaged,variants['soft_nms'],classes)
    print(stage,'PASSED: finite outputs, full frame coverage, metrics saved.',flush=True)


def visuals(rows,averaged,predictions,classes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2
    for row in rows:
        vid=row['video']; labels=sorted({e['label'] for e in row['events']}); scores=np.asarray(averaged[vid])
        pred=next(x['events'] for x in predictions if x['video']==vid)
        fig,axes=plt.subplots(len(labels),1,figsize=(14,max(4,len(labels)*1.35)),sharex=True,squeeze=False)
        for ax,label in zip(axes[:,0],labels):
            ax.plot(np.arange(len(scores))/row['fps'],scores[:,classes[label]],linewidth=0.8,color='#2455ad')
            for e in row['events']:
                if e['label']==label:ax.axvline(e['frame']/row['fps'],color='#209050',alpha=0.8)
            p=[e for e in pred if e['label']==label and e['score']>=0.2]
            ax.scatter([e['frame']/row['fps'] for e in p],[e['score'] for e in p],marker='x',color='#d05030',s=20)
            ax.set_ylabel(label,rotation=0,ha='right',fontsize=8);ax.set_ylim(0,1)
        axes[-1,0].set_xlabel('Seconds within downloaded clip')
        fig.suptitle(f'{vid}\nGreen: source labels mapped to clip; blue: scores; orange: soft-NMS ≥ 0.2')
        fig.tight_layout();fig.savefig(RESULTS/f'{vid}_timeline.png',dpi=120);plt.close(fig)
        picks=row['events'][:6]
        fig,axes=plt.subplots(2,3,figsize=(13,5))
        for ax in axes.flat:ax.axis('off')
        for ax,e in zip(axes.flat,picks):
            frame=row['_source_info']['start_frame']+e['frame']
            path=FRAMES/vid.split('_')[0]/f'{frame:06d}.jpg'
            im=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            ax.imshow(im);ax.set_title(f"{e['label']} @ {e['frame']}\nscore={scores[e['frame'],classes[e['label']]]:.3f}",fontsize=9)
        fig.tight_layout();fig.savefig(RESULTS/f'{vid}_frames.jpg',dpi=110);plt.close(fig)


def training(model,cfg,classes):
    seed();module=model._model;assert module.do_auxiliar_supervision
    rows=json.loads((ROOT/'validation/train_labels.json').read_text())
    assert len(rows)==2, 'Run download_subset.py to completion first'
    frames=[];labels=[];windows=[]
    reader=FrameReaderVideo(str(FRAMES),'finegym')
    for row in rows:
        target=row['events'][len(row['events'])//2]['frame']
        start=max(0,min(target-50,row['num_frames']-100))
        frames.append(reader.load_frames(row['video'],start,start+100,source_info=row['_source_info']))
        y=torch.zeros(100,dtype=torch.long)
        for e in row['events']:
            if start<=e['frame']<start+100:y[e['frame']-start]=classes[e['label']]
        assert y.count_nonzero()>0;labels.append(y)
        windows.append({'video':row['video'],'start':start,'positive_frames':int(y.count_nonzero())})
    batch={'frame':torch.stack(frames),'label':torch.stack(labels)}
    batch['frame2']=batch['frame'].flip(0).clone();batch['label2']=batch['label'].flip(0).clone()
    grads={};handles=[]
    for name,p in module.named_parameters():
        def hook(g,name=name):
            grads[name]={'finite':bool(torch.isfinite(g).all()),'norm':float(g.float().norm())}
        handles.append(p.register_hook(hook))
    head_outputs=[]
    def forward_hook(mod,inputs,output):
        assert all(torch.isfinite(v).all() for v in output.values())
        head_outputs.append({k:list(v.shape) for k,v in output.items()})
    handles.append(module.register_forward_hook(forward_hook))
    optimizer=torch.optim.AdamW(module.parameters(),lr=cfg['training']['learning_rate'])
    before={name:p.detach().cpu().clone() for name,p in module.named_parameters()}
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    from unittest.mock import patch
    native_ce=torch.nn.functional.cross_entropy
    head_losses=[]
    def observe_ce(*args,**kwargs):
        value=native_ce(*args,**kwargs)
        assert torch.isfinite(value).all()
        head_losses.append(float(value.detach()))
        return value
    with patch('model.model.F.cross_entropy',side_effect=observe_ce):
        loss=model.epoch([batch],optimizer=optimizer,scaler=None)
    assert len(head_losses)==3
    memory=gpu_memory();assert math.isfinite(loss)
    for h in handles:h.remove()
    assert grads and all(g['finite'] for g in grads.values())
    groups={}
    for name,g in grads.items():
        group=name.split('.')[0];groups[group]=groups.get(group,0)+g['norm']**2
    groups={k:math.sqrt(v) for k,v in groups.items()}
    for group in ['lowres_backbone','highres_backbone','_pred_fine','_pred_fine_lowres','_pred_fine_highres']:
        assert groups[group]>0,(group,groups[group])
    updates={}
    for name,p in module.named_parameters():
        assert torch.isfinite(p).all(),name
        norm=float((p.detach().cpu()-before[name]).norm())
        group=name.split('.')[0];updates[group]=updates.get(group,0)+norm**2
    updates={k:math.sqrt(v) for k,v in updates.items()}
    assert all(updates[g]>0 for g in ['lowres_backbone','highres_backbone','_pred_fine'])
    report={'loss':loss,'head_losses':dict(zip(['main','highres','lowres'],head_losses)),'native_training_method':'AdaSpot.epoch','optimizer':'AdamW','optimizer_steps':1,
        'learning_rate':cfg['training']['learning_rate'],'batch_size':2,'clip_len':100,'mixup':True,
        'auxiliary_heads_present':True,'head_outputs':head_outputs,'gradient_groups':groups,
        'all_gradient_tensors_finite':True,'gradient_tensors':len(grads),
        'parameters_without_grad':[n for n,p in module.named_parameters() if n not in grads],
        'parameter_update_norms':updates,'all_updated_parameters_finite':True,
        'windows':windows,'seconds':time.perf_counter()-started,**memory,'checkpoint_saved':False}
    write('training.json',report);print(json.dumps(report,indent=2),flush=True)
