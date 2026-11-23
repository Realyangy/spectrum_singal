#!/usr/bin/env python3
"""WSL terminal: upload alarm images and receive model files. No inference or quantization."""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sqlite3
import sys
import threading
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("Invalid relative model path")
    if PurePosixPath(value).is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("Model path escapes repository")
    return value


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


class Agent:
    def __init__(self, cloud, node_id, model_dir, events_dir, state_dir=None):
        parsed = urlsplit(cloud)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("--cloud must be an HTTP(S) base address without a path")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", node_id):
            raise ValueError("Invalid node ID")
        self.cloud = cloud.rstrip("/")
        self.node_id = node_id
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.events_dir = Path(events_dir).expanduser().resolve()
        self.state_dir = Path(state_dir).expanduser().resolve() if state_dir else self.model_dir.parent / ".cloud-edge-agent"
        self.stop = threading.Event()
        self.opener = build_opener(ProxyHandler({}))
        self._progress_at = 0
        self._upload_retry = {}

    def request(self, path, body=None, data=None, headers=None, timeout=30):
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers = {"Content-Type": "application/json", **(headers or {})}
        request = Request(self.cloud + path, data=data, headers=headers or {})
        try:
            return self.opener.open(request, timeout=timeout)
        except HTTPError as exc:
            detail = exc.read(8192).decode("utf-8", errors="replace")
            raise RuntimeError("HTTP %s: %s" % (exc.code, detail)) from exc

    def json_request(self, path, body=None, **kwargs):
        with self.request(path, body, **kwargs) as response:
            return json.load(response)

    def heartbeat(self):
        return self.json_request("/api/edge/heartbeat", {"node_id": self.node_id,
            "model_dir": str(self.model_dir), "events_dir": str(self.events_dir),
            "python": sys.executable, "platform": platform.platform()})

    def submit_task(self, client_request_id, title="终端哨兵任务", description="请确认任务后选择模型，完成量化与下发。", scene_id="person-whitelist", task=None, metadata=None, input_ref=""):
        """Call from the terminal business app; reuse the request ID when retrying."""
        payload = {"node_id": self.node_id, "client_request_id": client_request_id,
                   "scene_id": scene_id, "title": title, "description": description, "input_ref": input_ref}
        if task is not None:
            payload["task"] = task
        if metadata is not None:
            payload["metadata"] = metadata
        return self.json_request("/api/edge/tasks", payload)

    def get_task(self, task_id):
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise ValueError("Invalid task ID")
        return self.json_request("/api/edge/tasks/" + task_id + "?" + urlencode({"node_id": self.node_id}))

    def report_task(self, task_id, status, message="", no_alerts=False):
        self.get_task(task_id)  # Validate ownership and identifier before posting.
        return self.json_request("/api/edge/tasks/" + task_id + "/progress", {
            "node_id": self.node_id, "status": status, "message": message, "no_alerts": no_alerts})

    def upload_task_event(self, task_id, path):
        """Upload only this task's explicit output file; never bind the whole historical directory."""
        return self.upload_task_result(task_id, path, self.events_dir)

    def upload_task_result(self, task_id, path, output_dir=None):
        """Upload an explicit result from the scene's output directory, including compressed video."""
        self.get_task(task_id)
        root = Path(output_dir).expanduser().resolve() if output_dir else self.events_dir
        path = Path(path).expanduser()
        if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
            raise ValueError("Task result must be a regular file inside the output directory")
        path = path.resolve()
        stat = path.stat()
        video = path.suffix.lower() in {".mp4", ".webm"}
        limit = 256 * 1024 if path.suffix.lower() == ".json" else 256 * 1024**2 if video else 16 * 1024**2
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".json", ".mp4", ".webm"} or not 0 < stat.st_size <= limit:
            raise ValueError("Invalid task result type or size")
        digest = digest_file(path)
        if (stat.st_size, stat.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError("Task result is still being written")
        query = urlencode({"node_id": self.node_id, "task_id": task_id, "path": path.relative_to(root).as_posix()})
        with path.open("rb") as stream:
            result = self.json_request("/api/edge/results?" + query, data=stream, headers={
                "Content-Type": "application/octet-stream", "Content-Length": str(stat.st_size), "X-Content-SHA256": digest}, timeout=120)
        if result.get("sha256") != digest or result.get("status") != "saved":
            raise ValueError("Task result receipt does not match local file")
        return result

    def heartbeat_loop(self):
        while not self.stop.wait(5):
            try:
                self.heartbeat()
            except Exception as exc:
                log("心跳暂未送达：" + str(exc))

    def progress(self, job, status, count, error="", force=False):
        if not force and time.monotonic() - self._progress_at < 2:
            return
        if job.get("task_id") and status != "error":
            task = self.get_task(job["task_id"])
            if task["status"] == "cancelled":
                raise InterruptedError("云端已取消该任务，停止接收")
        try:
            self.json_request("/api/edge/jobs/" + job["job_id"] + "/progress", {
                "node_id": self.node_id, "status": status, "bytes_received": count, "error": error})
            self._progress_at = time.monotonic()
        except Exception as exc:
            log("进度上报等待重试：" + str(exc))

    def target(self, relative):
        target = self.model_dir / relative_path(relative)
        if self.model_dir not in target.resolve().parents:
            raise ValueError("Model destination escapes repository, possibly through a symlink")
        return target

    def receive_job(self, job):
        identifier = job.get("job_id", "")
        if not re.fullmatch(r"[0-9a-f]{32}", identifier) or job.get("node_id") != self.node_id:
            raise ValueError("Invalid or mismatched delivery job")
        files = job.get("files")
        if not isinstance(files, list) or not 1 <= len(files) <= 32:
            raise ValueError("Invalid delivery file list")
        seen = set()
        for file in files:
            relative = relative_path(file["path"])
            if relative in seen:
                raise ValueError("Duplicate model path")
            seen.add(relative)
            self.target(relative)
            if isinstance(file["size"], bool) or not isinstance(file["size"], int) or not 0 < file["size"] <= 2 * 1024**3:
                raise ValueError("Invalid model size")
            if not re.fullmatch(r"[0-9a-f]{64}", file["sha256"]):
                raise ValueError("Invalid model hash")
            if not file["url"].startswith("/api/edge/jobs/" + identifier + "/files/") or "?" in file["url"] or "#" in file["url"]:
                raise ValueError("Invalid download URL")
        total = sum(file["size"] for file in files)
        if job.get("total_bytes") != total:
            raise ValueError("Manifest total does not match files")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        stage = self.model_dir / ".agent-staging" / identifier
        stage.mkdir(parents=True, exist_ok=True)
        count = 0
        staged = []
        changed = []
        try:
            self.progress(job, "downloading", 0, force=True)
            for file in files:
                if self.stop.is_set():
                    raise InterruptedError("接收已停止，原模型保持可用")
                target = self.target(file["path"])
                if target.is_file() and target.stat().st_size == file["size"] and digest_file(target) == file["sha256"]:
                    count += file["size"]
                    continue
                partial = stage / "downloads" / (str(files.index(file)) + ".part")
                partial.parent.mkdir(parents=True, exist_ok=True)
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > file["size"]:
                    partial.unlink()
                    offset = 0
                if shutil.disk_usage(stage).free < file["size"] - offset + 64 * 1024**2:
                    raise OSError("终端剩余磁盘空间不足，模型未替换")
                if offset < file["size"]:
                    headers = {"Range": "bytes=%d-" % offset} if offset else {}
                    with self.request(file["url"], headers=headers, timeout=30) as response:
                        if offset and response.status == 206:
                            expected_range = "bytes %d-%d/%d" % (offset, file["size"] - 1, file["size"])
                            if response.headers.get("Content-Range") != expected_range:
                                raise ValueError("Download range does not match manifest")
                        elif response.status == 200:
                            offset = 0
                        else:
                            raise ValueError("Unexpected model download response")
                        with partial.open("ab" if offset else "wb") as output:
                            while True:
                                if self.stop.is_set():
                                    raise InterruptedError("接收已停止，部分文件留待续传")
                                block = response.read(64 * 1024)
                                if not block:
                                    break
                                if offset + len(block) > file["size"]:
                                    raise ValueError("Download exceeds expected model size")
                                output.write(block)
                                offset += len(block)
                                self.progress(job, "downloading", count + offset)
                            output.flush()
                            os.fsync(output.fileno())
                self.progress(job, "verifying", count + offset, force=True)
                if partial.stat().st_size != file["size"] or digest_file(partial) != file["sha256"]:
                    partial.unlink(missing_ok=True)
                    raise ValueError("模型大小或 SHA-256 校验失败，原模型未替换")
                staged.append((partial, target))
                count += file["size"]
            # 全部下载并校验后才逐文件替换；普通异常时回滚已替换的文件。
            # 运行中的推理程序仍需由使用方在接收完成后重新加载。
            self.progress(job, "saving", count, force=True)
            for index, (partial, target) in enumerate(staged):
                self.target(target.relative_to(self.model_dir).as_posix())
                target.parent.mkdir(parents=True, exist_ok=True)
                backup = stage / ("backup-%d" % index)
                if target.exists():
                    os.replace(target, backup)
                changed.append((target, backup))
                os.replace(partial, target)
            receipt = [{key: file[key] for key in ("path", "size", "sha256")} for file in files]
        except BaseException as exc:
            for target, backup in reversed(changed):
                target.unlink(missing_ok=True)
                if backup.exists():
                    os.replace(backup, target)
            self.progress(job, "error", min(count, total), str(exc), force=True)
            raise
        # ACK 网络失败不回滚已验证的模型；下次先核验本机文件，再补 ACK。
        self.json_request("/api/edge/jobs/" + identifier + "/ack", {"node_id": self.node_id, "files": receipt})
        shutil.rmtree(stage)
        log(("算法部署配置" if job.get("kind") == "algorithm-deployment" else "模型文件") + "已保存并确认：" + str(self.model_dir) + "；尚未启动任务处理")

    def upload_events(self, limit=4):
        if not self.events_dir.is_dir():
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.state_dir / "uploads.sqlite3")) as db:
            db.execute("CREATE TABLE IF NOT EXISTS sent (path TEXT PRIMARY KEY, modified INTEGER, size INTEGER, sha TEXT)")
            count = 0
            for path in self.events_dir.rglob("*"):
                if self.stop.is_set() or count >= limit:
                    break
                if not path.is_file() or path.is_symlink() or self.events_dir not in path.resolve().parents:
                    continue
                if self._upload_retry.get(str(path), 0) > time.monotonic():
                    continue
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".json"} or path.stem.lower().endswith((".thumb", ".thumbnail")):
                    continue
                stat = path.stat()
                if not stat.st_size or time.time_ns() - stat.st_mtime_ns < 2_000_000_000:
                    continue
                if stat.st_size > (256 * 1024 if path.suffix.lower() == ".json" else 16 * 1024**2):
                    continue
                relative = relative_path(path.relative_to(self.events_dir).as_posix())
                old = db.execute("SELECT modified,size FROM sent WHERE path=?", (relative,)).fetchone()
                if old == (stat.st_mtime_ns, stat.st_size):
                    continue
                digest = digest_file(path)
                after = path.stat()
                if (stat.st_mtime_ns, stat.st_size) != (after.st_mtime_ns, after.st_size):
                    continue
                query = urlencode({"node_id": self.node_id, "path": relative})
                count += 1
                try:
                    with path.open("rb") as stream:
                        result = self.json_request("/api/edge/events?" + query, data=stream, headers={
                            "Content-Type": "application/octet-stream", "Content-Length": str(stat.st_size), "X-Content-SHA256": digest})
                    if result.get("sha256") != digest or result.get("status") != "saved":
                        raise ValueError("云端上传确认与本地文件不一致")
                except Exception as exc:
                    self._upload_retry[str(path)] = time.monotonic() + 60
                    log("告警文件保留，稍后重试 " + relative + "：" + str(exc))
                    continue
                self._upload_retry.pop(str(path), None)
                db.execute("INSERT OR REPLACE INTO sent VALUES (?,?,?,?)", (relative, stat.st_mtime_ns, stat.st_size, digest))
                db.commit()
                log("告警文件已上传：" + relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", required=True, help="例如 http://127.0.0.1:18091")
    parser.add_argument("--node-id", default="lenovo-wsl")
    parser.add_argument("--model-dir", default="/home/lenovo/cloud-edge/model_repository")
    parser.add_argument("--events-dir", default="/home/lenovo/cloud-edge/person_whitelist_qr_demo/runs/events")
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--upload-events", action="store_true", help="上传告警目录中已有和新增的图片及 JSON；保留本地源文件")
    parser.add_argument("--check", action="store_true", help="仅检查云端健康，不上传文件、不领取模型")
    args = parser.parse_args()
    agent = Agent(args.cloud, args.node_id, args.model_dir, args.events_dir, args.state_dir)
    if args.check:
        result = agent.json_request("/api/health")
        if result.get("status") != "ok":
            raise RuntimeError("Unexpected health response")
        nodes = agent.json_request("/api/edge/nodes")
        if not isinstance(nodes.get("nodes"), list):
            raise RuntimeError("终端接口不可用，请连接联调版服务（默认端口 8091）")
        log("云端连接正常；未上传或下载文件")
        return
    agent.heartbeat()
    threading.Thread(target=agent.heartbeat_loop, daemon=True).start()
    log("终端已登记：" + args.node_id + "；模型保存目录：" + str(agent.model_dir))
    try:
        while not agent.stop.is_set():
            try:
                query = urlencode({"node_id": agent.node_id, "pending": "1"})
                jobs = agent.json_request("/api/edge/jobs?" + query)["jobs"]
                for job in reversed(jobs):
                    try:
                        agent.receive_job(job)
                    except Exception as exc:
                        log("模型任务保留，等待重试：" + str(exc))
                if args.upload_events:
                    agent.upload_events()
            except Exception as exc:
                log("任务暂未完成，5 秒后重试：" + str(exc))
            agent.stop.wait(5)
    except KeyboardInterrupt:
        agent.stop.set()
        log("终端接入程序已停止")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(str(error))
        sys.exit(1)
