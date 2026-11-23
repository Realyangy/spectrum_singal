"""Local RF task service. The existing cloud-edge agent receives the models."""
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import threading
import tempfile
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'frontend'
RUNS = DEMO / 'runs'
MODEL_ROOT = Path(os.environ.get('RF_MODEL_ROOT', '/home/lenovo/cloud-edge/model_repository')).resolve()
RECEIVED_MODELS = ROOT / 'received-models'
RUN_ROOTS = {'signal-separation': ROOT / 'spectrum-separation' / 'interactions', 'spectrum-analysis': ROOT / 'spectrum-recognition' / 'interactions'}
CLOUD = os.environ.get('RF_CLOUD_URL', 'http://127.0.0.1:18091').rstrip('/')
NODE = os.environ.get('RF_NODE_ID', 'lenovo-wsl')
PYTHON = os.environ.get('RF_SPECTRUM_PYTHON', '/home/lenovo/anaconda3/envs/rf-spectrum/bin/python')
OPENER = build_opener(ProxyHandler({}))
LOCK = threading.RLock()
STATE = {'phase': 'idle', 'active': False, 'message': '等待发起任务'}

def update(**kw):
    with LOCK:
        STATE.update(kw)

def api(path, body=None, headers=None):
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = Request(CLOUD + path, data=data, headers=headers or {'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=15) as response:
        return json.load(response)

def retry(action, stop):
    while not stop.is_set():
        try:
            return action()
        except HTTPError as exc:
            if exc.code not in (408, 429) and exc.code < 500:
                raise RuntimeError('云端 HTTP %s: %s' % (exc.code, exc.read(2048).decode(errors='replace')))
            update(message='云端暂时不可用，正在重试')
        except (URLError, TimeoutError, OSError):
            update(message='云端连接中断，正在重试')
        stop.wait(2)
    raise InterruptedError('本机任务已停止')

def choose_sample(task, sample_id):
    if task not in RUN_ROOTS or not isinstance(sample_id, str):
        raise ValueError('不支持的任务或样本编号')
    library = (ROOT / ('spectrum-separation' if task == 'signal-separation' else 'spectrum-recognition') / 'sample-library').resolve()
    manifest = json.loads((library / 'manifest.json').read_text(encoding='utf-8'))
    matches = [row for row in manifest['samples'] if row['id'] == sample_id]
    if len(matches) != 1:
        raise ValueError('未找到样本')
    item = matches[0]
    def resolve_file(name):
        path = (library / name).resolve()
        if library not in path.parents or not path.is_file():
            raise ValueError('样本文件不存在或路径无效')
        return str(path)
    metadata = {'task': task, 'sample_id': sample_id}
    if task == 'signal-separation':
        metadata.update(soi_type=item['soi_type'], interference=item['interference'])
    return {'metadata': metadata, 'input': resolve_file(item['input']),
            'truth': resolve_file(item['truth']) if task == 'signal-separation' else None}


def received_model_candidates(task):
    """Only include qmodels recorded by local RF tasks; ignore the shared repository's other models."""
    if task not in RUN_ROOTS:
        raise ValueError('不支持的任务类型')
    references = {}
    for kind, directory in RUN_ROOTS.items():
        for info in directory.glob('*/model-info.json'):
            try:
                data = json.loads(info.read_text(encoding='utf-8'))
                for item in data.get('files', []):
                    relative, digest = item.get('path', ''), item.get('sha256', '')
                    if not isinstance(relative, str) or not isinstance(digest, str):
                        continue
                    if not re.fullmatch(r'quantized/[a-f0-9]{64}\.qmodel', relative) or not re.fullmatch(r'[a-f0-9]{64}', digest):
                        continue
                    for path in (MODEL_ROOT / relative, RECEIVED_MODELS / (digest + '.qmodel')):
                        references.setdefault(path, {}).setdefault(digest, set()).add(kind)
            except (OSError, ValueError, TypeError, AttributeError):
                continue
    selected = []
    for path, hashes in references.items():
        if len(hashes) != 1:
            continue
        digest, owners = next(iter(hashes.items()))
        if owners != {task}:
            continue
        try:
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode) and path.parent.resolve() in (MODEL_ROOT / 'quantized', RECEIVED_MODELS.resolve()):
                selected.append((path, digest, metadata.st_size))
        except FileNotFoundError:
            pass
    return selected


def clear_received_models(task):
    removed = total = skipped = 0
    for path, expected, _ in received_model_candidates(task):
        try:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                skipped += 1; continue
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            after = path.lstat()
            if digest.hexdigest() != expected or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                skipped += 1; continue
            path.unlink()
            removed += 1; total += before.st_size
        except OSError:
            skipped += 1
    return {'removed': removed, 'bytes': total, 'skipped': skipped}


def verified_model(job):
    files = job.get('files') or []
    models = []
    for item in files:
        path = (MODEL_ROOT / item['path']).resolve()
        if MODEL_ROOT not in path.parents or not path.is_file() or path.stat().st_size != item['size']:
            raise ValueError('模型文件缺失或大小不符')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != item['sha256']:
            raise ValueError('模型文件 SHA-256 校验失败')
        if path.suffix == '.qmodel':
            models.append(path)
    if len(models) != 1:
        raise ValueError('本任务应收到一个量化模型包')
    source = models[0]
    expected = next(item['sha256'] for item in files if (MODEL_ROOT / item['path']).resolve() == source)
    RECEIVED_MODELS.mkdir(parents=True, exist_ok=True)
    target = RECEIVED_MODELS / (expected + '.qmodel')
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        return target
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=RECEIVED_MODELS, prefix='.receiving-', delete=False) as dest:
            temporary = Path(dest.name)
            digest = hashlib.sha256()
            with source.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block); dest.write(block)
        if digest.hexdigest() != expected: raise ValueError('模型复制时哈希不匹配')
        temporary.replace(target)
    finally:
        if temporary: temporary.unlink(missing_ok=True)
    return target

def upload(task_id, path, stop):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    query = urlencode({'node_id': NODE, 'task_id': task_id, 'path': path.name})
    def send():
        with path.open('rb') as stream:
            req = Request(CLOUD + '/api/edge/results?' + query, data=stream,
                          headers={'Content-Type':'application/octet-stream', 'Content-Length':str(path.stat().st_size), 'X-Content-SHA256':digest})
            with OPENER.open(req, timeout=60) as response:
                return json.load(response)
    response = retry(send, stop)
    if response.get('status') != 'saved' or response.get('sha256') != digest:
        raise ValueError('云端结果回执或哈希不匹配')

def compatible_spectrum_input(generated, expected):
    import numpy as np
    if generated.shape != (512, 512, 3) or expected.shape != generated.shape or generated.dtype != np.uint8 or expected.dtype != np.uint8:
        return False
    delta = np.abs(generated.astype(np.int16) - expected.astype(np.int16))
    return np.max(delta) <= 1 and np.count_nonzero(delta) <= delta.size * 0.02

def run(sample, stop, existing_task_id=None):
    task_id = None; process = None; prep = None
    try:
        nodes = retry(lambda: api('/api/edge/nodes'), stop)
        if not any(n.get('node_id') == NODE and n.get('online') for n in nodes.get('nodes', [])):
            raise RuntimeError('接入 Agent 未在线，请先启动现有接入 Agent')
        meta = sample['metadata']; kind = meta['task']
        update(phase='preparing', message='正在构造模型输入')
        prep = tempfile.TemporaryDirectory(prefix='rf-prepare-')
        prep_dir = Path(prep.name)
        library = ROOT / ('spectrum-separation' if kind == 'signal-separation' else 'spectrum-recognition') / 'sample-library'
        manifest = json.loads((library / 'manifest.json').read_text(encoding='utf-8'))
        info = next(row for row in manifest['samples'] if row['id'] == meta['sample_id'])
        prep_input = prep_dir / 'input-iq.npy'; shutil.copyfile(sample['input'], prep_input)
        if hashlib.sha256(prep_input.read_bytes()).hexdigest() != info['input_sha256']:
            raise ValueError('样本 IQ 哈希不匹配')
        prepared_sample = dict(sample, input=str(prep_input))
        if sample['truth']:
            prep_truth = prep_dir / 'truth.npz'; shutil.copyfile(sample['truth'], prep_truth)
            if hashlib.sha256(prep_truth.read_bytes()).hexdigest() != info['truth_sha256']:
                raise ValueError('样本真值哈希不匹配')
            prepared_sample['truth'] = str(prep_truth)
        prep_sample = prep_dir / 'sample.json'
        prep_sample.write_text(json.dumps(prepared_sample, ensure_ascii=False), encoding='utf-8')
        prep_log = prep_dir / 'prepare.log'
        with prep_log.open('w', encoding='utf-8') as log:
            process = subprocess.Popen([PYTHON, str(ROOT / 'backend/prepare.py'), str(prep_sample), str(prep_dir / 'model-input')], stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if stop.wait(.2): raise InterruptedError('任务已停止')
        if process.returncode: raise RuntimeError('构造模型输入失败：' + prep_log.read_text(errors='replace')[-1800:])
        prepared_file = prep_dir / 'model-input' / ('wavenet.npy' if kind == 'signal-separation' else 'yolo.npy')
        if hashlib.sha256(prepared_file.read_bytes()).hexdigest() != info['model_input_sha256']:
            if kind != 'spectrum-analysis':
                raise ValueError('本次模型输入与固定样本库不一致')
            # FFT roundoff may shift a few pixels by one; retain the verified fixed input.
            import numpy as np
            import cv2
            canonical = (library / info['model_input']).resolve()
            if library.resolve() not in canonical.parents or not canonical.is_file() or hashlib.sha256(canonical.read_bytes()).hexdigest() != info['model_input_sha256']:
                raise ValueError('固定模型输入文件哈希不匹配')
            generated, expected = np.load(prepared_file, allow_pickle=False), np.load(canonical, allow_pickle=False)
            if not compatible_spectrum_input(generated, expected):
                raise ValueError('本次频谱模型输入与固定样本库差异过大')
            shutil.copyfile(canonical, prepared_file)
            if not cv2.imwrite(str(prepared_file.with_suffix('.png')), expected):
                raise RuntimeError('无法保存固定频谱模型输入图')
        process = None
        payload = {'node_id': NODE, 'client_request_id': 'rf-' + uuid.uuid4().hex, 'scene_id':'rf-sepedge',
                   'task': kind, 'title': '信号分离' if kind == 'signal-separation' else '频谱检测与识别',
                   'description': '申请量化模型并处理本地测试样本', 'input_ref': sample['input'], 'metadata': meta}
        task = (retry(lambda: api('/api/edge/tasks/' + existing_task_id + '?' + urlencode({'node_id':NODE})), stop)
                if existing_task_id else retry(lambda: api('/api/edge/tasks', payload), stop))
        task_id = task.get('task_id')
        if not isinstance(task_id, str) or not re.fullmatch('[0-9a-f]{32}', task_id) or task.get('node_id') != NODE:
            raise ValueError('云端返回的任务编号或设备无效')
        if existing_task_id:
            context = task.get('context') or {}
            if task_id != existing_task_id or task.get('scene_id') != 'rf-sepedge' or context.get('task') != kind or context.get('metadata') != meta or context.get('input_ref') != sample['input']:
                raise ValueError('本地样本与云端原任务不一致，不能恢复')
            if task.get('status') in ('completed', 'failed', 'cancelled'):
                raise ValueError('云端任务已结束，不能恢复')
        out = RUN_ROOTS[kind] / task_id; out.mkdir(parents=True, exist_ok=bool(existing_task_id))
        shutil.copytree(prep_dir / 'model-input', out / 'model-input', dirs_exist_ok=True)
        RUNS.mkdir(parents=True, exist_ok=True)
        link = RUNS / task_id
        if not link.exists(): link.symlink_to(out, target_is_directory=True)
        elif link.resolve() != out.resolve(): raise ValueError('任务记录路径冲突')
        sample_file = out / 'sample.json'; sample_file.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding='utf-8')
        (out / 'sample-info.json').write_text(json.dumps({'task_id':task_id, 'sample':info}, ensure_ascii=False, indent=2), encoding='utf-8')
        raw = out / 'input'; raw.mkdir(exist_ok=True)
        input_copy = raw / 'input-iq.npy'
        shutil.copyfile(prep_input, input_copy)
        worker_sample = dict(sample, input=str(input_copy))
        if sample['truth']:
            truth_copy = raw / 'truth.npz'; shutil.copyfile(prep_truth, truth_copy)
            worker_sample['truth'] = str(truth_copy)
        worker_sample_file = out / 'worker-sample.json'
        worker_sample_file.write_text(json.dumps(worker_sample, ensure_ascii=False, indent=2), encoding='utf-8')
        update(task_id=task_id, phase='waiting', message='等待云端确认、量化和下发')
        def poll():
            current = retry(lambda: api('/api/edge/tasks/' + task_id + '?' + urlencode({'node_id':NODE})), stop)
            if current.get('task_id') != task_id or current.get('node_id') != NODE:
                raise ValueError('云端任务归属不匹配')
            if current.get('status') in ('cancelled', 'failed'):
                raise InterruptedError(current.get('message') or '云端任务已取消或失败')
            return current
        while not stop.is_set():
            current = poll(); status = current.get('status'); job_id = current.get('job_id') or (current.get('delivery') or {}).get('job_id')
            snapshot = current.get('snapshot') or {}
            if status == 'quantization_failed':
                reason = (snapshot.get('quant_job_data') or {}).get('error') or '未知原因'
                message = '云侧量化失败，等待云侧重试：' + reason
            else:
                message = current.get('message') or str(status)
            update(message=message, model=snapshot.get('model'), quantization=snapshot.get('bits'))
            if job_id:
                jobs = retry(lambda: api('/api/edge/jobs?' + urlencode({'node_id':NODE})), stop).get('jobs', [])
                job = next((x for x in jobs if x.get('job_id') == job_id), None)
                if job:
                    if job.get('task_id') != task_id or job.get('node_id') != NODE or job.get('kind') != 'quantized-weights':
                        raise ValueError('模型下发任务归属或类型不匹配')
                    total = job.get('total_bytes') or 0; received = (job.get('progress') or {}).get('bytes_received', 0)
                    update(phase='receiving', receive_progress=min(1, received/total) if total else 0, job_id=job_id)
                    if job.get('status') == 'saved' and status in ('model_ready', 'processing'):
                        break
            stop.wait(2)
        if stop.is_set(): raise InterruptedError('任务已停止')
        update(phase='loading', message='校验本次下发模型', receive_progress=1)
        model = verified_model(job)
        (out / 'model-info.json').write_text(json.dumps({'model_path': str(model), 'files': job['files'], 'task_id': task_id}, ensure_ascii=False, indent=2), encoding='utf-8')
        poll()
        # The worker reports loading and inference as a single operation; processing is only
        # reported after model validation in the worker (via its ready marker).
        marker = out / 'model-ready'; marker.unlink(missing_ok=True)
        env = dict(os.environ, RF_READY_MARKER=str(marker), MPLCONFIGDIR='/tmp/rf-edge-mpl', YOLO_CONFIG_DIR='/tmp/rf-edge-yolo')
        log = (out / 'worker.log').open('w', encoding='utf-8')
        process = subprocess.Popen([PYTHON, str(ROOT / 'backend/infer.py'), str(worker_sample_file), str(model), str(out)], stdout=log, stderr=subprocess.STDOUT, env=env)
        reported = False
        while process.poll() is None:
            poll()
            if marker.exists() and not reported:
                retry(lambda: api('/api/edge/tasks/' + task_id + '/progress', {'node_id':NODE, 'status':'processing', 'message':'模型已加载，开始处理样本'}), stop)
                reported = True; update(phase='processing', message='模型已加载，正在处理样本')
            stop.wait(2)
        log.close()
        if process.returncode:
            raise RuntimeError('推理失败：' + (out / 'worker.log').read_text(errors='replace')[-1800:])
        if not reported:
            if not marker.exists(): raise RuntimeError('模型未报告加载成功')
            poll(); retry(lambda: api('/api/edge/tasks/' + task_id + '/progress', {'node_id':NODE, 'status':'processing', 'message':'模型已加载，开始处理样本'}), stop)
        poll()
        result = json.loads((out / 'result.json').read_text(encoding='utf-8'))
        update(phase='uploading', message='推理完成，正在回传本次结果', result=result)
        for name in ['result.json'] + result['images']:
            poll(); upload(task_id, out / name, stop)
        poll()
        retry(lambda: api('/api/edge/tasks/' + task_id + '/progress', {'node_id':NODE, 'status':'completed', 'message':'推理及结果回传完成'}), stop)
        (out / 'status.json').write_text(json.dumps({'status':'completed','task_id':task_id},ensure_ascii=False),encoding='utf-8')
        update(phase='completed', active=False, message='处理完成，结果已回传', result=result)
    except Exception as exc:
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        if task_id and (not isinstance(exc, InterruptedError) or stop.is_set()):
            if stop.is_set():
                marker_dir = MODEL_ROOT.parent / '.cloud-edge-agent' / 'cancelled_tasks'
                marker_dir.mkdir(parents=True, exist_ok=True)
                (marker_dir / task_id).write_text('端侧已停止任务', encoding='utf-8')
            try: api('/api/edge/tasks/' + task_id + '/progress', {'node_id':NODE, 'status':'failed', 'message':str(exc)[:1000]})
            except Exception: pass
        if task_id and 'out' in locals() and out.is_dir():
            (out / 'status.json').write_text(json.dumps({'status':'cancelled' if isinstance(exc, InterruptedError) else 'failed','task_id':task_id,'message':str(exc)},ensure_ascii=False),encoding='utf-8')
        update(phase='cancelled' if isinstance(exc, InterruptedError) else 'error', active=False, message=str(exc))
    finally:
        if prep: prep.cleanup()

STOP = threading.Event()
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs): super().__init__(*args, directory=str(DEMO), **kwargs)
    def response_json(self, value, status=200):
        payload = json.dumps(value, ensure_ascii=False).encode(); self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length',str(len(payload))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(payload)
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/':
            self.send_response(302); self.send_header('Location', '/live/'); self.end_headers()
        elif path == '/api/rf/status':
            with LOCK: self.response_json(dict(STATE))
        elif path in ('/api/rf/models/signal-separation', '/api/rf/models/spectrum-analysis'):
            task = path.rsplit('/', 1)[-1]
            try:
                with LOCK:
                    models = received_model_candidates(task)
                    self.response_json({'files': len(models), 'bytes': sum(item[2] for item in models), 'active': bool(STATE.get('active'))})
            except (OSError, ValueError) as exc: self.response_json({'error': str(exc)}, 500)
        elif path in ('/api/rf/samples/signal-separation', '/api/rf/samples/spectrum-analysis'):
            task = path.rsplit('/', 1)[-1]
            try: self.response_json(json.loads((ROOT / ('spectrum-separation' if task == 'signal-separation' else 'spectrum-recognition') / 'sample-library' / 'manifest.json').read_text(encoding='utf-8')))
            except (OSError, ValueError) as exc: self.response_json({'error': str(exc)}, 500)
        else: super().do_GET()
    def do_POST(self):
        if urlsplit(self.path).path == '/api/rf/cancel':
            with LOCK:
                if not STATE.get('active'): return self.response_json({'error':'没有运行中的任务'}, 409)
                STOP.set(); STATE['message'] = '正在停止任务'; return self.response_json(dict(STATE))
        endpoint = urlsplit(self.path).path
        if endpoint == '/api/rf/models/delete':
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096: raise ValueError('请求无效')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict): raise ValueError('请求无效')
                with LOCK:
                    if STATE.get('active'): return self.response_json({'error': '任务运行中，暂不能删除模型'}, 409)
                    return self.response_json(clear_received_models(body.get('task')))
            except (ValueError, TypeError, OSError) as exc: return self.response_json({'error': str(exc)}, 400)
        if endpoint not in ('/api/rf/start', '/api/rf/resume'): return self.send_error(404)
        try:
            length = int(self.headers.get('Content-Length','0'))
            if length <= 0 or length > 4096: raise ValueError('请求无效')
            body = json.loads(self.rfile.read(length))
            existing = body.get('task_id') if endpoint == '/api/rf/resume' else None
            if existing:
                if not isinstance(existing, str) or not re.fullmatch('[0-9a-f]{32}', existing): raise ValueError('任务编号无效')
                prior = json.loads((RUNS / existing / 'sample.json').read_text(encoding='utf-8'))
                sample = choose_sample(prior['metadata']['task'], prior['metadata']['sample_id'])
                if sample != prior: raise ValueError('本地样本记录发生变化，无法恢复')
            elif endpoint == '/api/rf/resume':
                raise ValueError('请提供任务编号')
            else:
                sample = choose_sample(body.get('task'), body.get('sample_id'))
            with LOCK:
                if STATE.get('active'): return self.response_json({'error':'已有任务正在运行'},409)
                STOP.clear(); STATE.clear(); STATE.update(active=True, phase='requesting', message='正在恢复任务' if existing else '正在提交云端任务', sample_id=sample['metadata']['sample_id'], task_type=sample['metadata']['task'])
                threading.Thread(target=run, args=(sample, STOP, existing), daemon=True).start()
                self.response_json(dict(STATE))
        except (ValueError, KeyError, TypeError, OSError) as exc: self.response_json({'error':str(exc)},400)

if __name__ == '__main__':
    RUNS.mkdir(parents=True, exist_ok=True)
    print('频谱端侧服务：http://127.0.0.1:8080/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', int(os.environ.get('RF_PORT', '8080'))), Handler).serve_forever()
