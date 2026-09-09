#!/usr/bin/env python3
"""Reproducible real-module validation; run from any working directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('HF_HOME', str(ROOT / 'artifacts/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'artifacts/torch'))
import torch
from huggingface_hub import HfApi, hf_hub_download
from inference import dict_to_namespace
from model.model import AdaSpot
from util.dataset import load_classes


def save(name, value):
    p = ROOT / 'validation/results' / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + '\n')


def load_model(model_name):
    cfg = json.loads((ROOT / 'config' / model_name.split('_')[0] / f'{model_name}.json').read_text())
    args = dict_to_namespace(cfg['model'])
    args.clip_len = cfg['data']['clip_len']
    args.dataset = cfg['data']['dataset']
    args.num_classes = cfg['data']['num_classes']
    classes = load_classes(f'data/{args.dataset}/class.txt')
    revision_file = ROOT / f'validation/{model_name}.json'
    if revision_file.exists():
        source = json.loads(revision_file.read_text())
        assert source['model'] == model_name
    else:
        source = {'repo': 'arturxe/AdaSpot', 'revision': HfApi().model_info('arturxe/AdaSpot').sha,
                  'model': model_name, 'file': f'{model_name}-1/checkpoint_best.pt'}
        revision_file.write_text(json.dumps(source, indent=2) + '\n')
    checkpoint = hf_hub_download(source['repo'], source['file'], revision=source['revision'])
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = AdaSpot(device='cuda', args_model=args, args_training=dict_to_namespace(cfg['training']), classes=classes)
    result = model._model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    # Check actual tensor values, not only matching names and shapes.
    actual = model._model.state_dict()
    assert all(torch.equal(actual[k].cpu(), v) for k, v in state.items())
    save(f'{model_name}_checkpoint.json', {**source, 'sha256': digest, 'bytes': Path(checkpoint).stat().st_size,
         'parameters': sum(p.numel() for p in model._model.parameters()), 'state_tensors': len(state),
         'strict_load': True, 'all_tensors_equal': True, 'missing_keys': [], 'unexpected_keys': [],
         'module_type': type(model._model).__qualname__, 'config': cfg})
    print('Strict loading and equality of all checkpoint tensors passed.', flush=True)
    return model, cfg, classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['checkpoint', 'smoke', 'evaluate', 'train'])
    p.add_argument('--model', default='FineGym_big')
    args = p.parse_args()
    torch.set_num_threads(8)
    model, cfg, classes = load_model(args.model)
    if args.stage != 'checkpoint':
        from validation_checks import inference, training
        if args.stage == 'train':
            training(model, cfg, classes)
            source=json.loads((ROOT / f'validation/{args.model}.json').read_text())
            path=hf_hub_download(source['repo'],source['file'],revision=source['revision'],local_files_only=True)
            original=json.loads((ROOT/f'validation/results/{args.model}_checkpoint.json').read_text())
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==original['sha256']
            report=json.loads((ROOT/'validation/results/training.json').read_text())
            report['checkpoint_sha256_after']=original['sha256']
            report['checkpoint_unchanged']=True
            save('training.json',report)
        else:
            inference(model, cfg, classes, args.stage)

if __name__ == '__main__':
    main()
