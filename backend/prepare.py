"""Prepare the first model input for a selected IQ sample before cloud request."""
import json
import sys
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'spectrum-recognition/src'))

def main(sample_path, output_path):
    sample=json.loads(Path(sample_path).read_text(encoding='utf-8'))
    task=sample['metadata']['task']
    iq=np.load(sample['input'],allow_pickle=False)
    shape=(40960,) if task=='signal-separation' else (262144,)
    if task not in ('signal-separation','spectrum-analysis') or iq.shape!=shape or not np.iscomplexobj(iq) or not np.isfinite(iq).all():
        raise ValueError('输入 IQ 格式错误')
    out=Path(output_path);out.mkdir(parents=True,exist_ok=True)
    if task=='signal-separation':
        np.save(out/'wavenet.npy',np.stack((iq.real,iq.imag))[None].astype(np.float32),allow_pickle=False)
    else:
        import cv2
        from torchsig.transforms.functional import spectrogram
        from yolo_inference import to_yolo_image
        image=to_yolo_image(np.asarray(spectrogram(iq,512,512),dtype=np.float32))
        np.save(out/'yolo.npy',image,allow_pickle=False)
        if not cv2.imwrite(str(out/'yolo.png'),image): raise RuntimeError('无法保存 YOLO 输入')
if __name__=='__main__': main(sys.argv[1],sys.argv[2])
