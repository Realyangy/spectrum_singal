"""Demodulate a single recorded sample in the separate Sionna environment."""
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'spectrum-separation' / 'src'))
import rfcutils

def calculate(soi, mixed_path, prediction_path, truth_path):
    truth = np.load(str(truth_path), allow_pickle=False)
    target = truth['target_iq'].reshape(1, -1).astype(np.complex64)
    bits = truth['true_bits'].reshape(1, -1)
    mixed = np.load(str(mixed_path), allow_pickle=False).reshape(1, -1).astype(np.complex64)
    predicted = np.load(str(prediction_path), allow_pickle=False).reshape(1, -1).astype(np.complex64)
    if soi not in ('QPSK', 'OFDMQPSK') or target.shape != (1, 40960) or mixed.shape != target.shape or predicted.shape != target.shape:
        raise ValueError('Invalid sample or signal type')
    grid = rfcutils.get_resource_grid(512) if soi == 'OFDMQPSK' else None
    def decode(signal):
        x = tf.convert_to_tensor(signal, dtype=tf.complex64)
        decoded, _ = rfcutils.qpsk_matched_filter_demod(x) if grid is None else rfcutils.ofdm_demod(x, grid)
        result = decoded.numpy()
        if result.shape != bits.shape or not np.isin(result, [0, 1]).all():
            raise ValueError('Invalid demodulator output')
        return result
    if np.count_nonzero(decode(target) != bits):
        raise ValueError('Clean reference does not reproduce its ground truth bits')
    input_errors = int(np.count_nonzero(decode(mixed) != bits))
    output_errors = int(np.count_nonzero(decode(predicted) != bits))
    return {'bit_count': int(bits.size), 'input_bit_errors': input_errors,
            'output_bit_errors': output_errors, 'input_ber': input_errors / bits.size,
            'output_ber': output_errors / bits.size}

if __name__ == '__main__':
    Path(sys.argv[5]).write_text(json.dumps(calculate(*sys.argv[1:5]), ensure_ascii=False), encoding='utf-8')
