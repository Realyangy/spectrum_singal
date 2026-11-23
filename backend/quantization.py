"""Portable weight-only post-training quantization for trusted local checkpoints."""

import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path


FORMAT_VERSION = 1
QUANT_MODES = ("INT8", "INT4", "FP16", "混合精度 (INT8/FP16)")


class QuantizationError(Exception):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise QuantizationError("真实量化需要 PyTorch；当前服务环境没有安装 torch") from exc
    return torch


def _load_state(torch, path):
    try:
        loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        if path.suffix.lower() != ".pt":
            raise QuantizationError("无法安全读取模型权重：%s" % path.name) from exc
        try:
            from ultralytics import YOLO
            loaded = YOLO(str(path)).model.state_dict()
        except ImportError as import_exc:
            raise QuantizationError("该 YOLO 权重需要安装 ultralytics 才能读取") from import_exc
        except Exception as load_exc:
            raise QuantizationError("无法读取 YOLO 权重：%s" % load_exc) from load_exc

    if hasattr(loaded, "state_dict"):
        loaded = loaded.state_dict()
    if isinstance(loaded, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = loaded.get(key)
            if hasattr(value, "state_dict"):
                loaded = value.state_dict()
                break
            if isinstance(value, dict) and value and all(hasattr(v, "shape") for v in value.values()):
                loaded = value
                break
    if not isinstance(loaded, dict) or not loaded:
        raise QuantizationError("模型文件中没有可量化的 state_dict：%s" % path.name)
    return loaded


def _bits_for(mode, tensor):
    if mode == "INT8":
        return "int8"
    if mode == "INT4":
        return "int4"
    if mode == "FP16":
        return "fp16"
    return "int8" if tensor.ndim >= 2 else "fp16"


def _quantize_tensor(torch, tensor, encoding):
    source = tensor.detach().to(device="cpu").contiguous()
    if encoding == "fp16":
        return {"encoding": "fp16", "tensor": source.to(dtype=torch.float16)}, source.numel()

    values = source.to(dtype=torch.float32)
    limit = 127 if encoding == "int8" else 7
    maximum = float(values.abs().max().item()) if values.numel() else 0.0
    scale = maximum / limit if maximum else 1.0
    quantized = torch.round(values / scale).clamp(-limit, limit).to(dtype=torch.int16).reshape(-1)
    if encoding == "int8":
        return {"encoding": "int8", "scale": scale, "tensor": quantized.to(dtype=torch.int8).reshape(source.shape)}, source.numel()

    unsigned = (quantized & 0x0F).to(dtype=torch.uint8)
    if unsigned.numel() % 2:
        unsigned = torch.cat((unsigned, torch.zeros(1, dtype=torch.uint8)))
    packed = unsigned[0::2] | (unsigned[1::2] << 4)
    return {"encoding": "int4", "scale": scale, "elements": source.numel(), "tensor": packed}, source.numel()


def _dequantize_entry(torch, entry, shape):
    encoding, tensor = entry["encoding"], entry["tensor"]
    if encoding == "fp16":
        return tensor.to(dtype=torch.float32).reshape(shape)
    if encoding == "int8":
        return tensor.to(dtype=torch.float32).mul(entry["scale"]).reshape(shape)
    packed = tensor.to(dtype=torch.int16)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    values = torch.stack((low, high), dim=1).reshape(-1)[:entry["elements"]]
    values = torch.where(values >= 8, values - 16, values)
    return values.to(dtype=torch.float32).mul(entry["scale"]).reshape(shape)


def quantize_checkpoint(sources, mode, output_path, on_progress=None):
    """Write an atomic .qmodel archive; inputs are never modified or removed."""
    if mode not in QUANT_MODES:
        raise QuantizationError("不支持的量化方式：%s" % mode)
    paths = [Path(source).resolve(strict=True) for source in sources]
    if not paths or any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise QuantizationError("模型权重不存在或为空")

    torch = _torch()
    source_hashes = [sha256_file(path) for path in paths]
    manifest = {"format": "cloud-console-qmodel", "version": FORMAT_VERSION,
                "method": "weight-only-post-training", "mode": mode, "models": []}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".quant-", suffix=".tmp", dir=output_path.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary_name, "w", compression=zipfile.ZIP_STORED) as archive:
            for model_index, source in enumerate(paths):
                state = _load_state(torch, source)
                quantized_state, tensor_info = {}, []
                total = len(state)
                for index, (name, tensor) in enumerate(state.items()):
                    if not hasattr(tensor, "is_floating_point") or not tensor.is_floating_point():
                        quantized_state[name] = {"encoding": "raw", "tensor": tensor.detach().cpu() if hasattr(tensor, "detach") else tensor}
                        continue
                    encoding = _bits_for(mode, tensor)
                    entry, count = _quantize_tensor(torch, tensor, encoding)
                    quantized_state[name] = entry
                    tensor_info.append({"name": str(name), "shape": list(tensor.shape), "encoding": encoding, "elements": count})
                    if on_progress:
                        on_progress(int((model_index + (index + 1) / max(1, total)) / len(paths) * 90))

                payload = io.BytesIO()
                torch.save(quantized_state, payload)
                member = "weights/%02d.pt" % model_index
                archive.writestr(member, payload.getvalue())
                if sha256_file(source) != source_hashes[model_index]:
                    raise QuantizationError("量化期间源模型发生变化，请重试")
                source_size = source.stat().st_size
                manifest["models"].append({
                    "source_name": source.name,
                    "source_sha256": source_hashes[model_index],
                    "source_bytes": source_size,
                    "member": member,
                    "tensor_count": len(tensor_info),
                    "tensors": tensor_info,
                })
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))

        os.replace(temporary_name, output_path)
        if on_progress:
            on_progress(100)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    return {"format": manifest["format"], "version": FORMAT_VERSION, "method": manifest["method"],
            "mode": mode, "source_bytes": sum(item["source_bytes"] for item in manifest["models"]),
            "artifact_bytes": output_path.stat().st_size, "artifact_name": output_path.name,
            "artifact_sha256": sha256_file(output_path), "model_count": len(manifest["models"]),
            "tensor_count": sum(item["tensor_count"] for item in manifest["models"])}


def inspect_qmodel(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    if manifest.get("format") != "cloud-console-qmodel" or manifest.get("version") != FORMAT_VERSION:
        raise QuantizationError("不支持的量化模型文件格式")
    return manifest


def dequantize_qmodel(path, model_index=0):
    """Load one qmodel member as float32 tensors for a model-specific runtime adapter."""
    torch = _torch()
    manifest = inspect_qmodel(path)
    try:
        model = manifest["models"][model_index]
        with zipfile.ZipFile(path) as archive:
            stream = io.BytesIO(archive.read(model["member"]))
        try:
            state = torch.load(stream, map_location="cpu", weights_only=True)
        except TypeError:
            stream.seek(0)
            state = torch.load(stream, map_location="cpu")
    except (IndexError, KeyError, zipfile.BadZipFile) as exc:
        raise QuantizationError("量化模型内容无效") from exc
    shapes = {row["name"]: row["shape"] for row in model["tensors"]}
    result = {}
    for name, entry in state.items():
        result[name] = (_dequantize_entry(torch, entry, shapes[name]) if entry["encoding"] != "raw"
                        else entry["tensor"])
    return result, manifest
