"""Run one task using only the delivered qmodel's parameters."""
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'spectrum-recognition/src'))
import numpy as np
import torch
from quantization import inspect_qmodel, dequantize_qmodel

def state_for(path, source_name):
    manifest = inspect_qmodel(path)
    indices = [i for i, item in enumerate(manifest['models']) if item['source_name'] == source_name]
    if len(indices) != 1:
        raise ValueError('云端模型包缺少或重复包含 ' + source_name)
    return dequantize_qmodel(path, indices[0])[0]

def separation(sample, model, output):
    from omegaconf import OmegaConf
    sys.path.insert(0, str(root / 'spectrum-separation' / 'rfchallenge'))
    from src.torchwavenet import Wave
    from src.config_torchwavenet import ModelConfig
    config = OmegaConf.load(root / 'spectrum-separation/rfchallenge/src/configs/wavenet.yml')
    net = Wave(ModelConfig(**OmegaConf.to_container(config.model, resolve=True)))
    net.load_state_dict(state_for(model, 'weights.pt'), strict=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device).eval()
    if os.environ.get('RF_READY_MARKER'):
        Path(os.environ['RF_READY_MARKER']).touch()
    iq = np.load(sample['input'], allow_pickle=False)
    if iq.shape != (40960,) or not np.iscomplexobj(iq) or not np.isfinite(iq).all():
        raise ValueError('输入 IQ 格式错误')
    model_inputs = output / 'model-input'; model_inputs.mkdir(exist_ok=True)
    prepared = model_inputs / 'wavenet.npy'
    array = np.load(prepared, allow_pickle=False) if prepared.is_file() else np.stack((iq.real, iq.imag))[None].astype(np.float32)
    if array.shape != (1,2,40960) or array.dtype != np.float32: raise ValueError('WaveNet 输入无效')
    np.save(prepared, array, allow_pickle=False)
    x = torch.from_numpy(array).to(device)
    with torch.inference_mode():
        y = net(x).cpu().numpy()[0]
    pred = (y[0] + 1j*y[1]).astype(np.complex64)
    if pred.shape != iq.shape or not np.isfinite(pred).all():
        raise ValueError('模型输出无效')
    np.save(output / 'recovered-iq.npy', pred)
    target = np.load(sample['truth'], allow_pickle=False)['target_iq'].reshape(-1)
    mse = float(np.mean(np.abs(pred - target)**2))
    ber_path = output / 'ber.json'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
    subprocess.run([os.environ.get('RF_SIONNA_PYTHON', '/home/lenovo/anaconda3/envs/rf-sionna/bin/python'),
                    str(root / 'backend/ber.py'), sample['metadata']['soi_type'], sample['input'],
                    str(output / 'recovered-iq.npy'), sample['truth'], str(ber_path)], check=True, env=env)
    ber = json.loads(ber_path.read_text())
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(12, 7), constrained_layout=True)
    for axis, label, data in zip(ax, ['Mixed input', 'Recovered', 'Reference'], [iq, pred, target]):
        axis.plot(np.real(data[:1200]), linewidth=.7); axis.set_title(label); axis.grid(alpha=.2)
    fig.savefig(output / 'waveforms.png', dpi=120); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), constrained_layout=True)
    for axis, label, data in zip(axes, ['Mixed input', 'Recovered', 'Reference'], [iq, pred, target]):
        axis.specgram(data, NFFT=256, Fs=1.0, noverlap=192); axis.set_title(label)
    fig.savefig(output / 'spectrograms.png', dpi=120); plt.close(fig)
    return {'task': 'signal-separation', 'sample_id': sample['metadata']['sample_id'],
            'mse': mse, 'mse_db': float(10*np.log10(max(mse, 1e-30))), **ber,
            'images': ['waveforms.png', 'spectrograms.png'], 'output_iq': 'recovered-iq.npy'}

def spectrum(sample, model, output):
    import cv2
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from torchsig.transforms.functional import spectrogram
    from xcit_model import XCiTClassifier
    from yolo_xcit_pipeline import extract_iq, classify_iq, draw_overlay
    from yolo_inference import box_features, to_yolo_image
    raw = np.load(sample['input'], allow_pickle=False)
    if raw.shape != (262144,) or not np.iscomplexobj(raw) or not np.isfinite(raw).all():
        raise ValueError('输入 IQ 格式错误')
    model_inputs = output / 'model-input'; model_inputs.mkdir(exist_ok=True)
    prepared = model_inputs / 'yolo.npy'
    image = np.load(prepared, allow_pickle=False) if prepared.is_file() else to_yolo_image(np.asarray(spectrogram(raw, 512, 512), dtype=np.float32))
    if image.shape[:2] != (512,512) or not np.isfinite(image).all(): raise ValueError('YOLO 输入无效')
    np.save(prepared, image, allow_pickle=False)
    if not cv2.imwrite(str(model_inputs / 'yolo.png'), image): raise RuntimeError('无法保存 YOLO 输入')
    detector = YOLO('yolo11n.yaml', task='detect')
    # The wideband detector has one class, while the standard YAML defaults to 80.
    detector.model = DetectionModel(detector.model.yaml, nc=1, verbose=False)
    detector.model.load_state_dict(state_for(model, 'yolo11n_wideband_v1.0.0.pt'), strict=True)
    classifier = XCiTClassifier(input_channels=2, num_classes=57, xcit_version='tiny_12_p16_224', ds_method='downsample', ds_rate=16)
    classifier.load_state_dict(state_for(model, 'xcit_wideband_adapted.ckpt'), strict=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    classifier.to(device).eval()
    if os.environ.get('RF_READY_MARKER'):
        Path(os.environ['RF_READY_MARKER']).touch()
    detected = detector.predict(source=image, imgsz=512, conf=.25, device=device, verbose=False)[0]
    predictions = []
    for i, (coords, confidence) in enumerate(zip(detected.boxes.xyxy.cpu().numpy(), detected.boxes.conf.cpu().numpy()), 1):
        box = [float(coords[0]/image.shape[1]), float(coords[1]/image.shape[0]), float(coords[2]/image.shape[1]), float(coords[3]/image.shape[0])]
        record = {'detection_id': i, 'confidence': float(confidence), 'box_xyxy_normalized': box,
                  'features': box_features(box, 10000000., len(raw)/10000000.)}
        try:
            extracted, quality = extract_iq(raw, box, 10000000.)
            np.save(model_inputs / f'xcit-{i:03d}.npy', extracted, allow_pickle=False)
            record['extraction'] = quality
            record['xcit'] = classify_iq(classifier, extracted, device)
        except (ValueError, RuntimeError) as exc:
            record['xcit_error'] = str(exc)
        predictions.append(record)
    draw_overlay(image, predictions, output / 'detections.png')
    if not (output / 'detections.png').is_file():
        raise RuntimeError('未能生成检测图片')
    return {'task': 'spectrum-analysis', 'sample_id': sample['metadata']['sample_id'],
            'sample_rate_hz': 10000000, 'detections': predictions, 'images': ['detections.png']}

if __name__ == '__main__':
    sample = json.loads(Path(sys.argv[1]).read_text())
    model, output = Path(sys.argv[2]), Path(sys.argv[3])
    output.mkdir(parents=True, exist_ok=True)
    result = separation(sample, model, output) if sample['metadata']['task'] == 'signal-separation' else spectrum(sample, model, output)
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
