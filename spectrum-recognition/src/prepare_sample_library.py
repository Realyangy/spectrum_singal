#!/usr/bin/env python3
"""Generate 100 fixed, numbered TorchSig wideband IQ samples."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from benchmark_combined import make_stream
from yolo_inference import as_signal_list

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'sample-library'
SEED = 20261111

def main():
    OUT.mkdir(exist_ok=False)
    stream, capture, meta, cfg = make_stream(ROOT/'configs/wideband_clean_val_all.yaml', SEED)
    rows=[]
    for i in range(100):
        spec, labels = next(stream)[0]
        iq = capture['raw']
        if iq.shape != (262144,) or iq.dtype != np.complex64 or not np.isfinite(iq).all(): raise ValueError('Invalid generated IQ')
        boxes = as_signal_list(labels[0], 'box')
        names = as_signal_list(labels[1], 'name')
        if len(boxes) != len(names): raise ValueError('Labels mismatch')
        d=OUT / f'{i+1:03d}'; d.mkdir()
        p=d/'input-iq.npy'; np.save(p, iq, allow_pickle=False)
        rows.append({'id':f'spectrum-{i+1:03d}', 'sample_number':i+1, 'source_index':i, 'input':str(p.relative_to(OUT)), 'sample_rate_hz':float(meta['sample_rate']), 'ground_truth':{'boxes':[np.asarray(b).tolist() for b in boxes], 'class_names':[str(n) for n in names]}, 'input_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        if (i+1)%10==0: print('generated',i+1,flush=True)
    (OUT/'manifest.json').write_text(json.dumps({'seed':SEED,'torchsig_version':'2.1.1','samples':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved',len(rows),'samples',flush=True)
if __name__ == '__main__': main()
