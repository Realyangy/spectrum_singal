import collections, importlib.util, json, math, os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "separation"
spec = importlib.util.spec_from_file_location("rf_ber", str(ROOT / "backend" / "ber.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
manifest = json.loads((ROOT / "spectrum-separation/sample-library/manifest.json").read_text(encoding="utf-8"))
expected_ids = {row["id"] for row in manifest["samples"]}
records = []
with (OUT / "predictions.jsonl").open(encoding="utf-8") as src, (OUT / "per_sample.jsonl").open("w", encoding="utf-8") as dest:
    for line in src:
        row = json.loads(line)
        row.update(module.calculate(row["soi_type"], row["input"], row["prediction"], row["truth"]))
        dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        records.append(row)
        if len(records) % 10 == 0:
            dest.flush()
            print("SEPARATION_BER {}/{}".format(len(records), len(expected_ids)), flush=True)
ids = [r["sample_id"] for r in records]
if len(ids) != len(expected_ids) or set(ids) != expected_ids:
    raise ValueError("Sample coverage mismatch or duplicate IDs")
def summarize(items):
    n = len(items)
    bits = sum(r["bit_count"] for r in items)
    input_err = sum(r["input_bit_errors"] for r in items)
    output_err = sum(r["output_bit_errors"] for r in items)
    avg_mse = sum(r["mse"] for r in items)/n
    return {"samples":n, "bit_count":bits, "input_bit_errors":input_err,
            "output_bit_errors":output_err, "input_ber":input_err/bits, "output_ber":output_err/bits,
            "mean_input_mse":sum(r["input_mse"] for r in items)/n,
            "mean_mse":avg_mse, "mean_mse_db":10*math.log10(max(avg_mse,1e-30))}
groups = collections.defaultdict(list)
sinr = collections.defaultdict(list)
for row in records:
    groups[(row["soi_type"], row["interference"])].append(row)
    sinr[(row["soi_type"], row["interference"], row["nominal_sinr_db"])].append(row)
summary = {"model_type":"original WaveNet", "seed":manifest.get("seed"), "overall":summarize(records),
           "groups":[dict({"soi_type":a,"interference":b}, **summarize(items)) for (a,b),items in sorted(groups.items())],
           "by_sinr":[dict({"soi_type":a,"interference":b,"nominal_sinr_db":c}, **summarize(items)) for (a,b,c),items in sorted(sinr.items())]}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print("SEPARATION_DONE {}".format(len(records)), flush=True)
