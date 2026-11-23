import collections, json, shutil, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "spectrum-recognition"
OUT = Path(__file__).resolve().parent / "spectrum"
sys.path.insert(0, str(SPEC / "src"))
from xcit_model import XCiTClassifier
from yolo_xcit_pipeline import extract_iq, classify_iq, signal_family
from yolo_inference import evaluate, xywh_to_xyxy

library = SPEC / "sample-library"
manifest = json.loads((library / "manifest.json").read_text(encoding="utf-8"))
rows = manifest["samples"]
OUT.mkdir(parents=True)
yolo_path = SPEC / "models/yolo11n_wideband_v1.0.0.pt"
xcit_path = SPEC / "models/xcit_wideband_adapted.ckpt"
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in rf-spectrum")
detector = YOLO(str(yolo_path))
classifier = XCiTClassifier.load_from_checkpoint(str(xcit_path), map_location="cpu", weights_only=False).to("cuda:0").eval()
totals = collections.Counter()
class_totals = collections.defaultdict(collections.Counter)
data = OUT / "yolo_dataset"
images = data / "images/val"
labels = data / "labels/val"
images.mkdir(parents=True)
labels.mkdir(parents=True)
with (OUT / "per_sample.jsonl").open("w", encoding="utf-8") as stream:
    for number, row in enumerate(rows, 1):
        iq = np.load(library / row["input"], allow_pickle=False)
        image_path = library / row["model_input"]
        image = np.load(image_path, allow_pickle=False)
        gt_info = row["ground_truth"]
        if len(gt_info["boxes"]) != len(gt_info["class_names"]):
            raise ValueError("Ground truth mismatch: " + row["id"])
        gt = []
        lines = []
        for raw, name in zip(gt_info["boxes"], gt_info["class_names"]):
            _, cx, cy, w, h = map(float, raw)
            gt.append({"true_class":name, "box_xyxy_normalized":xywh_to_xyxy(cx, cy, w, h)})
            lines.append(f"0 {cx:.9f} {cy:.9f} {w:.9f} {h:.9f}")
            class_totals[name]["gt"] += 1
        stem = f"{number:03d}"
        png_path = image_path.with_name("yolo.png")
        png = cv2.imread(str(png_path))
        if png is None or not np.array_equal(png, image):
            raise ValueError("Saved model input and PNG differ: " + row["id"])
        shutil.copyfile(png_path, images / (stem + ".png"))
        (labels / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        result = detector.predict(source=image, imgsz=512, conf=0.25, device=0, verbose=False)[0]
        preds = []
        for coords, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            box = [float(coords[0]/image.shape[1]), float(coords[1]/image.shape[0]), float(coords[2]/image.shape[1]), float(coords[3]/image.shape[0])]
            pred = {"box_xyxy_normalized":box, "confidence":float(score)}
            try:
                extracted, _ = extract_iq(iq, box, float(row["sample_rate_hz"]))
                pred["xcit"] = classify_iq(classifier, extracted, "cuda:0")
            except (ValueError, RuntimeError) as exc:
                pred["xcit_error"] = str(exc)
                totals["extraction_or_classification_errors"] += 1
            preds.append(pred)
        matched = evaluate(preds, gt, 0.5)
        totals["gt"] += len(gt)
        totals["predictions"] += len(preds)
        totals["tp"] += matched["TP"]
        totals["fp"] += matched["FP"]
        totals["fn"] += matched["FN"]
        for match in matched["matches"]:
            actual = gt[match["ground_truth_index"]]["true_class"]
            classified = preds[match["prediction_index"]].get("xcit")
            if classified is None:
                continue
            totals["matched_classified"] += 1
            class_totals[actual]["matched_classified"] += 1
            if classified["fine_top1"] == actual:
                totals["fine_top1_correct"] += 1
                class_totals[actual]["fine_top1_correct"] += 1
            if actual in [p["class"] for p in classified["fine_top5"]]:
                totals["fine_top5_correct"] += 1
            if classified["family_top1"] == signal_family(actual):
                totals["family_correct"] += 1
        stream.write(json.dumps({"sample_id":row["id"], "ground_truth":gt, "predictions":preds, "matching":matched}, ensure_ascii=False, allow_nan=False) + "\n")
        if number % 5 == 0 or number == len(rows):
            stream.flush()
            print(f"SPECTRUM {number}/{len(rows)} | GT {totals['gt']} TP {totals['tp']}", flush=True)
count = totals["matched_classified"]
gt_count = totals["gt"]
tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
ratio = lambda a,b: a/b if b else None
summary = {"sample_count":len(rows), "seed":manifest.get("seed"), "model_type":"original weights",
           "yolo_model":str(yolo_path), "xcit_model":str(xcit_path), "device":torch.cuda.get_device_name(0),
           "confidence_threshold":0.25, "matching_iou":0.5, "counts":dict(totals),
           "detection_precision":ratio(tp,tp+fp), "detection_recall":ratio(tp,tp+fn),
           "matched_fine_top1_accuracy":ratio(totals["fine_top1_correct"],count),
           "matched_fine_top5_accuracy":ratio(totals["fine_top5_correct"],count),
           "matched_family_accuracy":ratio(totals["family_correct"],count),
           "end_to_end_fine_top1_recall":ratio(totals["fine_top1_correct"],gt_count),
           "end_to_end_family_recall":ratio(totals["family_correct"],gt_count),
           "per_class":{name:dict(values) for name,values in sorted(class_totals.items())}}
def save():
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
save()
config_path = data / "dataset.yaml"
config_path.write_text(yaml.safe_dump({"path":str(data.resolve()),"train":"images/val","val":"images/val","nc":1,"names":{0:"signal"}}, sort_keys=False), encoding="utf-8")
print("SPECTRUM_FIXED_CONF_DONE; starting AP validation", flush=True)
metrics = detector.val(data=str(config_path), split="val", imgsz=512, batch=8, device=0, conf=0.001, iou=0.7, workers=0, single_cls=True, verbose=False, plots=False, project=str(OUT), name="yolo_validation", exist_ok=True)
summary["detection_map50"] = float(metrics.box.map50)
summary["detection_map50_95"] = float(metrics.box.map)
summary["map_confidence_threshold"] = 0.001
summary["map_nms_iou"] = 0.7
save()
print(f"SPECTRUM_DONE {len(rows)} mAP50={summary['detection_map50']:.4f}", flush=True)
