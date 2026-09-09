#!/usr/bin/env python3
"""Recheck persisted evidence, real input bytes, predictions, and measured results."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from util.score import compute_mAPs
R=ROOT/'validation/results'
FRAMES=Path(os.environ.get('ADASPOT_FRAME_DIR','/tmp/adaspot-finegym-frames'))

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=read(ROOT/'validation/subset.json')
    checkpoint=read(R/'FineGym_big_checkpoint.json')
    environment=read(R/'environment.json')
    train=read(R/'training.json')
    test_rows=read(ROOT/'validation/test_labels.json')
    train_rows=read(ROOT/'validation/train_labels.json')
    assert len(test_rows)==20 and len(train_rows)==2 and len(manifest['clips'])==22
    assert not {r['video'].split('_E_')[0] for r in test_rows}&{r['video'].split('_E_')[0] for r in train_rows}
    assert len({e['label'] for r in test_rows for e in r['events']})==32
    actual_data=read(R/'data.json')
    by_id={r['video']:r for r in test_rows+train_rows}
    frame_reports={r['video']:r for r in actual_data['clips']}
    for item in manifest['clips']:
        vid=item['original']['video'];row=by_id[vid]
        assert sha(ROOT/f'artifacts/finegym/videos/{vid}.mp4')==item['sha256']
        original=next(r for r in read(ROOT/f"data/finegym/{item['split']}.json") if r['video']==vid)
        assert original==item['original'],vid
        old_origin=original['_source_info']['start_frame']-original['_source_info']['pad'][0]
        new_origin=row['_source_info']['start_frame']
        assert len(original['events'])==len(row['events'])
        for old,new in zip(original['events'],row['events']):
            assert old['label']==new['label'] and old_origin+old['frame']==new_origin+new['frame']
            assert 0<=new['frame']<row['num_frames']
        digest=hashlib.sha256()
        for i in range(row['num_frames']):
            p=FRAMES/vid.split('_')[0]/f'{new_origin+i:06d}.jpg'
            digest.update(p.read_bytes())
        assert digest.hexdigest()==frame_reports[vid]['frames_sha256']
    for stage,n in [('smoke',2),('evaluate',20)]:
        metrics=read(R/f'{stage}_metrics.json');pred=read(R/f'{stage}_predictions.json')
        rows=test_rows[:n];assert metrics['num_clips']==n
        assert metrics['num_frames']==sum(r['num_frames'] for r in rows)
        correct=total=0
        classes={label:i+1 for i,label in enumerate((ROOT/'data/finegym/class.txt').read_text().splitlines())}
        for row in rows:
            cache=np.load(ROOT/f"artifacts/scores/{row['video']}.npz")
            scores=cache['scores'];support=cache['support']
            assert scores.shape==(row['num_frames'],33)
            assert np.isfinite(scores).all() and (support>0).all()
            labels=np.zeros(len(scores),dtype=np.int64)
            for e in row['events']:labels[e['frame']]=classes[e['label']]
            correct+=int((scores.argmax(1)==labels).sum());total+=len(labels)
        assert abs(correct/total-metrics['frame_accuracy'])<1e-12
        for variant in ['raw','nms','soft_nms']:
            maps,tols=compute_mAPs(rows,pred[variant])
            assert all(abs(float(v)-metrics[variant]['mAP'][str(t)])<1e-12 for v,t in zip(maps,tols))
        for rel,digest in metrics['provenance']['source_hashes'].items():assert sha(ROOT/rel)==digest
        assert metrics['provenance']['checkpoint_sha256']==checkpoint['sha256']
        assert all(t['peak_allocated_bytes']<environment['gpu_bytes'] for t in metrics['timings'])
    assert checkpoint['strict_load'] and checkpoint['all_tensors_equal'] and checkpoint['parameters']==15369355
    assert not checkpoint['missing_keys'] and not checkpoint['unexpected_keys']
    assert train['optimizer_steps']==1 and train['auxiliary_heads_present']
    assert math.isfinite(train['loss']) and all(math.isfinite(x) for x in train['head_losses'].values())
    assert train['all_gradient_tensors_finite'] and train['all_updated_parameters_finite']
    assert train['checkpoint_sha256_after']==checkpoint['sha256'] and train['checkpoint_unchanged']
    assert train['peak_allocated_bytes']<environment['gpu_bytes']
    for g in ['lowres_backbone','highres_backbone','_pred_fine','_pred_fine_lowres','_pred_fine_highres']:
        assert math.isfinite(train['gradient_groups'][g]) and train['gradient_groups'][g]>0
        assert math.isfinite(train['parameter_update_norms'][g]) and train['parameter_update_norms'][g]>0
    for row in test_rows[:2]:
        for suffix in ['_timeline.png','_frames.jpg']:
            with Image.open(R/(row['video']+suffix)) as im:im.verify()
    report={'passed':True,'verified_inputs':22,'verified_eval_clips':20,'verified_classes':32,
        'checks':['input video SHA-256','extracted frame SHA-256','original labels and absolute event frame conservation',
          'source-video train/test disjointness','finite fully covered scores','frame accuracy recomputation',
          'native mAP recomputation','native source hashes','strict full-module weight loading evidence',
          'finite native main and auxiliary losses','gradient and parameter update evidence','VRAM fit','readable visualizations'],
        'limitation':'Archive trim origins are inferred from integer-second event IDs, not independently matched to full source video frames. Metrics describe this transformed subset, not the original full benchmark.'}
    (R/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
