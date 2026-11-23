import collections, json, sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "separation"
SEP = ROOT / "signal-separation"
sys.path.insert(0, str(SEP / "rfchallenge"))
from src.torchwavenet import Wave
from src.config_torchwavenet import ModelConfig

library = SEP / "sample-library"
manifest = json.loads((library / "manifest.json").read_text(encoding="utf-8"))
rows = manifest["samples"]
groups = collections.defaultdict(list)
for row in rows:
    groups[(row["soi_type"], row["interference"])].append(row)
predictions = OUT / "predictions"
predictions.mkdir(parents=True)
cfg = ModelConfig(**OmegaConf.to_container(OmegaConf.load(SEP / "rfchallenge/src/configs/wavenet.yml").model, resolve=True))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in rf-spectrum")
count = 0
with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as stream:
    for (soi, interference), members in sorted(groups.items()):
        model_path = SEP / "pretrained-models/WaveNet" / f"dataset_{soi.lower()}_{interference.lower()}_mixture_wavenet" / "weights.pt"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model = Wave(cfg)
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        model.to("cuda:0").eval()
        for start in range(0, len(members), 4):
            subset = members[start:start+4]
            inputs = np.stack([np.load(library / row["input"], allow_pickle=False) for row in subset])
            x = np.stack((inputs.real, inputs.imag), axis=1).astype(np.float32)
            with torch.inference_mode():
                result = model(torch.from_numpy(x).to("cuda:0")).cpu().numpy()
            for row, iq, y in zip(subset, inputs, result):
                estimate = (y[0] + 1j * y[1]).astype(np.complex64)
                if estimate.shape != (40960,) or not np.isfinite(estimate).all():
                    raise ValueError("Invalid WaveNet prediction: " + row["id"])
                truth = np.load(library / row["truth"], allow_pickle=False)["target_iq"].reshape(-1)
                pred_path = predictions / (row["id"] + ".npy")
                np.save(pred_path, estimate, allow_pickle=False)
                record = {"sample_id":row["id"], "soi_type":soi, "interference":interference,
                          "nominal_sinr_db":row["nominal_sinr_db"], "model":str(model_path),
                          "input":str(library / row["input"]), "truth":str(library / row["truth"]),
                          "prediction":str(pred_path), "input_mse":float(np.mean(np.abs(iq-truth)**2)),
                          "mse":float(np.mean(np.abs(estimate-truth)**2))}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
            stream.flush()
            print(f"SEPARATION_INFERENCE {count}/{len(rows)}", flush=True)
        del model
print(f"SEPARATION_INFERENCE_DONE {count}", flush=True)
