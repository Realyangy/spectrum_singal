#!/usr/bin/env python3
"""Export 50 aligned saved Seed-42 records per supported signal pair."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'sample-library'
SOURCES = {'CommSignal2': 'mit_sionna_seed42', 'CommSignal5G1': 'commsignal5g1_seed42', 'EMISignal1': 'emisignal1_seed42'}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    records = []
    for interference, experiment in SOURCES.items():
        base = ROOT / 'output' / experiment
        summary = json.loads((base / 'summary.json').read_text())
        sinrs, n = summary['sinrs_db'], summary['samples_per_sinr']
        if len(sinrs) != 7 or n != 10: raise ValueError('Unexpected source SINR layout')
        indices = [group*n+i for group in range(7) for i in range(7)] + [7]
        for soi in ('QPSK', 'OFDMQPSK'):
            prefix = f'TestSet1Example_testmixture_{soi}_{interference}'
            source = np.load(base / 'dataset' / (prefix+'.npy'), allow_pickle=False, mmap_mode='r')
            # These pickles are locally generated trusted experiment files.
            with (base / 'dataset' / f'GroundTruth_TestSet1Example_Dataset_{soi}_{interference}.pkl').open('rb') as f:
                mixed, target, bits = pickle.load(f)
            if source.shape != (70,40960) or target.shape != source.shape or mixed.shape != source.shape or bits.shape[0] != 70: raise ValueError('Unexpected source data shape')
            group = OUT / soi / interference; group.mkdir(parents=True)
            for no, idx in enumerate(indices, 1):
                if not np.array_equal(source[idx],mixed[idx]): raise ValueError(f'Input/truth mismatch {soi} {interference} {idx}')
                d = group / f'{no:03d}'; d.mkdir()
                x, t = d/'input-iq.npy', d/'truth.npz'
                np.save(x, np.asarray(source[idx], dtype=np.complex64), allow_pickle=False)
                np.savez(t, target_iq=np.asarray(target[idx],dtype=np.complex64), true_bits=np.asarray(bits[idx]))
                records.append({'id':f'{soi.lower()}-{interference.lower()}-{no:03d}', 'soi_type':soi, 'interference':interference, 'sample_number':no, 'source_index':idx, 'nominal_sinr_db':sinrs[idx//n], 'input':str(x.relative_to(OUT)), 'truth':str(t.relative_to(OUT)), 'input_sha256':digest(x), 'truth_sha256':digest(t)})
            print(soi, interference, len(indices), flush=True)
    (OUT/'manifest.json').write_text(json.dumps({'seed':42,'samples':records},ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved',len(records),'samples',flush=True)
if __name__ == '__main__': main()
