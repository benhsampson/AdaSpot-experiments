#!/usr/bin/env python3
"""Select and fetch 20 test + 2 train FineGym clips with HTTP ZIP ranges."""
import copy
import hashlib
import json
import random
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import cv2
cv2.setNumThreads(4)
from remote_zip import RemoteZipFile
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'validation/subset.json'
BASE = ROOT / 'artifacts/finegym'
FRAMES = Path(os.environ.get('ADASPOT_FRAME_DIR', '/tmp/adaspot-finegym-frames'))

def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as r: return json.load(r)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def select():
    repo='HopLeeTop/Sports-QA'
    revision=get_json(f'https://huggingface.co/api/datasets/{repo}')['sha']
    entries=get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}')
    archives=[]; available={}
    for e in entries:
        if not e['path'].startswith('finegym_'): continue
        a={'url':f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{e["path"]}',
           'size':e['size'],'archive_sha256':e['lfs']['oid']}
        archives.append(a)
        with zipfile.ZipFile(RemoteZipFile(a['url'],a['size'])) as z:
            for info in z.infolist():
                if info.filename.endswith('.mp4') and not info.filename.startswith('__MACOSX'):
                    available[Path(info.filename).stem]={'archive':len(archives)-1,'member':info.filename,
                        'bytes':info.file_size,'zip_crc32':info.CRC}
    rng=random.Random(1); chosen=[]
    for split,number in [('test',20),('train',2)]:
        rows=json.loads((ROOT/f'data/finegym/{split}.json').read_text())
        # Integer FPS avoids ambiguity in long-video timestamp rounding.
        rows=[r for r in rows if r['video'] in available and r['fps'] in (25.,30.)]
        rng.shuffle(rows); covered=set(); groups={g:0 for g in ['BB','FX','UB','VT']}
        for _ in range(number):
            eligible=[r for r in rows if split=='train' or groups[r['events'][0]['label'].split('_')[0]]<5]
            r=min(eligible,key=lambda r:(-len({e['label'] for e in r['events']}-covered),r['num_frames']))
            covered.update(e['label'] for e in r['events'])
            groups[r['events'][0]['label'].split('_')[0]]+=1
            rows.remove(r)
            chosen.append({'split':split,'original':r,**available[r['video']]})
    result={'dataset':'finegym','source_repo':repo,'source_revision':revision,'seed':1,
        'selection':'Integer 25/30 FPS; greedy unseen-class coverage, then shortest clips; 5 test clips per apparatus; seed-1 tie order',
        'archives':archives,'clips':chosen,
        'alignment':'Archive clip origin assumed to be integer start seconds in event ID; retain source-label absolute frame coordinates; report trimming limitation explicitly.'}
    MANIFEST.write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    manifest=json.loads(MANIFEST.read_text()) if MANIFEST.exists() else select()
    BASE.mkdir(parents=True,exist_ok=True)
    archives={}; transferred=0; reports=[]; adapted={'test':[],'train':[]}
    for item in manifest['clips']:
        row=item['original']; vid=row['video']; video=BASE/'videos'/f'{vid}.mp4'
        video.parent.mkdir(exist_ok=True)
        print('Preparing',item['split'],vid,flush=True)
        if not video.exists():
            ai=item['archive']; a=manifest['archives'][ai]
            if ai not in archives:
                remote=RemoteZipFile(a['url'],a['size']);archives[ai]=(remote,zipfile.ZipFile(remote))
            blob=archives[ai][1].read(item['member']) # checks ZIP CRC automatically
            assert len(blob)==item['bytes']
            temp=video.with_suffix('.partial');temp.write_bytes(blob);temp.replace(video)
        digest=sha(video)
        if 'sha256' in item: assert digest==item['sha256'],vid
        item['sha256']=digest
        cap=cv2.VideoCapture(str(video))
        fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert abs(fps-row['fps'])<1e-5,(vid,fps,row['fps'])
        start_seconds=int(vid.rsplit('_',2)[1]); end_seconds=int(vid.rsplit('_',2)[2])
        assert abs(count/fps-(end_seconds-start_seconds+1))<0.1,(vid,count/fps,start_seconds,end_seconds)
        origin=round(start_seconds*fps)
        original_origin=row['_source_info']['start_frame']-row['_source_info']['pad'][0]
        out=copy.deepcopy(row)
        out['num_frames']=count
        # Native FineGym reader uses source video ID + absolute frame filenames.
        out['_source_info']={'start_frame':origin,'end_frame':origin+count,'pad':[0,0],'effective_pad':[0,0]}
        for e in out['events']: e['frame']+=original_origin-origin
        assert all(0<=e['frame']<count for e in out['events']),(vid,out['events'])
        frame_dir=FRAMES/vid.split('_')[0];frame_dir.mkdir(parents=True,exist_ok=True)
        i=0; frame_hash=hashlib.sha256()
        while True:
            ok,bgr=cap.read()
            if not ok: break
            frame_path=frame_dir/f'{origin+i:06d}.jpg'
            if not frame_path.exists():
                assert cv2.imwrite(str(frame_path),cv2.resize(bgr,(796,448)),[cv2.IMWRITE_JPEG_QUALITY,95])
            decoded=cv2.imread(str(frame_path));assert decoded is not None and decoded.shape==(448,796,3)
            frame_hash.update(frame_path.read_bytes());i+=1
        cap.release();assert i==count,(vid,i,count)
        adapted[item['split']].append(out)
        evidence={'video':vid,'split':item['split'],'frames':count,'fps':fps,'sha256':digest,
            'frames_sha256':frame_hash.hexdigest(),'original_origin':original_origin,'download_origin':origin,
            'event_frame_shift':original_origin-origin,'events':len(out['events']),'all_frames_decoded':True}
        reports.append(evidence)
        print(json.dumps(evidence),flush=True)
        MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
        for split,rows in adapted.items():
            (BASE/f'{split}.json').write_text(json.dumps(rows,indent=2)+'\n')
            (ROOT/f'validation/{split}_labels.json').write_text(json.dumps(rows,indent=2)+'\n')
    test_ids={x['video'].split('_E_')[0] for x in adapted['test']}
    train_ids={x['video'].split('_E_')[0] for x in adapted['train']}
    assert not test_ids & train_ids
    assert len(adapted['test'])==20 and len(adapted['train'])==2
    covered=sorted({e['label'] for r in adapted['test'] for e in r['events']})
    transferred=sum(r.transferred for r,z in archives.values())
    assert sum(x['bytes'] for x in manifest['clips'])<50_000_000_000
    report={'clips':reports,'test_classes':covered,'test_class_count':len(covered),
        'retained_video_bytes':sum(x['bytes'] for x in manifest['clips']),
        'range_bytes_this_run':transferred,'source_video_split_disjoint':True}
    (ROOT/'validation/results/data.json').write_text(json.dumps(report,indent=2)+'\n')
    for split,rows in adapted.items():
        (ROOT/f'validation/{split}_labels.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('Data verification complete',len(covered),'classes;',transferred,'bytes transferred',flush=True)

if __name__=='__main__':main()
