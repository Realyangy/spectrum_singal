#!/usr/bin/env python3
"""Create and index the first-model input corresponding to each fixed sample."""
import hashlib
import json
from pathlib import Path
import tempfile
from prepare import main as prepare

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    with tempfile.TemporaryDirectory(prefix='rf-index-') as tmp:
        tmp=Path(tmp)
        for task in ('signal-separation','spectrum-analysis'):
            root=ROOT/('spectrum-separation' if task=='signal-separation' else 'spectrum-recognition')/'sample-library'; path=root/'manifest.json'
            data=json.loads(path.read_text(encoding='utf-8'))
            for n,item in enumerate(data['samples'],1):
                iq=(root/item['input']).resolve()
                if root.resolve() not in iq.parents or sha(iq)!=item['input_sha256']: raise ValueError('IQ hash mismatch')
                sample=tmp/'sample.json'; sample.write_text(json.dumps({'metadata':{'task':task},'input':str(iq)}))
                out=iq.parent/'model-input'
                if out.exists(): raise FileExistsError(str(out))
                prepare(sample,out)
                model_input=out/('wavenet.npy' if task=='signal-separation' else 'yolo.npy')
                item['model_input']=str(model_input.relative_to(root));item['model_input_sha256']=sha(model_input)
                if n%50==0: print(task,n,flush=True)
            path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print('prepared all fixed model inputs',flush=True)
if __name__=='__main__': main()
