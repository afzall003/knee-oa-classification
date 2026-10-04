"""Export the frozen KneeGrade pipeline for in-browser inference.

Writes into web/:
  models/<member>.bin, models/pipeline.json   weights (batch norm folded into convolutions) and config
  data/curves.json, data/examples.json        training curves and example list for the site
  assets/, examples/                          figures and example X-rays (copied from space/)
  parity/                                     NOT deployed: 826 validation X-rays + Python outputs,
                                              used by parity.html to prove the browser matches Python

Run from the project root with the knee-oa environment:  python export_for_web.py
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
WEB = PROJECT / "web"
MEMBERS = ["exp1_augmentation", "exp3_aug_wider"]
TRAIN_COUNTS = [2286, 1046, 1516, 757, 173]
GRADE_NAMES = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]


def fnv1a(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def export_member(model, name, out_dir):
    """Fold BN into each conv, write float32 weights; return the manifest entry."""
    import keras
    chunks, offset = [], 0

    def add(arr):
        nonlocal offset
        a = np.ascontiguousarray(arr, dtype="<f4")
        chunks.append(a.tobytes())
        spec = {"offset": offset, "shape": list(a.shape)}
        offset += a.size
        return spec

    convs = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)]
    assert len(convs) == 8, f"{name}: expected 8 conv layers, found {len(convs)}"
    layers = []
    for conv in convs:
        bn = model.get_layer(conv.name.replace("_conv", "_bn"))
        gamma, beta, mean, var = (w.astype(np.float64) for w in bn.get_weights())
        scale = gamma / np.sqrt(var + bn.epsilon)
        kernel = conv.get_weights()[0].astype(np.float64) * scale
        layers.append({"name": conv.name, "kernel": add(kernel), "bias": add(beta - mean * scale)})
    kernel, bias = model.get_layer("kl_grade").get_weights()
    entry = {"name": name, "file": f"{name}.bin", "params": int(model.count_params()),
             "convs": layers, "dense": {"kernel": add(kernel), "bias": add(bias)}}
    (out_dir / entry["file"]).write_bytes(b"".join(chunks))
    return entry


def main():
    import keras
    sys.path.insert(0, str(PROJECT / "src"))
    import preprocessing as pp

    for sub in ("models", "data", "parity"):
        (WEB / sub).mkdir(parents=True, exist_ok=True)

    # ---- Models ----
    models, entries = [], []
    for name in MEMBERS:
        model = keras.models.load_model(PROJECT / "models" / f"{name}.keras")
        models.append(model)
        entries.append(export_member(model, name, WEB / "models"))
        size = (WEB / "models" / f"{name}.bin").stat().st_size / 1e6
        print(f"Exported {name}: {entries[-1]['params']:,} parameters, {size:.1f} MB")
    pipeline = {"version": "1.0", "members": entries, "train_counts": TRAIN_COUNTS, "grade_names": GRADE_NAMES,
                "joint_band": [70, 154, 20, 204], "model_repo": "afzal2003/knee-oa-kl-grading"}
    (WEB / "models" / "pipeline.json").write_text(json.dumps(pipeline))

    # ---- Site data ----
    curves = {}
    for run in MEMBERS:
        path = PROJECT / "reports" / f"{run}_history.csv"
        with path.open() as f:
            rows = list(csv.DictReader(f))
        curves[run] = {k: [float(r[k]) for r in rows] for k in rows[0]}
    (WEB / "data" / "curves.json").write_text(json.dumps(curves))

    for folder in ("assets", "examples"):
        src = PROJECT / "space" / folder
        if src.exists():
            shutil.copytree(src, WEB / folder, dirs_exist_ok=True)
    for stale in (WEB / "assets").glob("*_history.csv"):
        stale.unlink()
    examples = [{"url": f"examples/{p.name}", "name": p.name, "label": int(p.name[5])}
                for p in sorted((WEB / "examples").glob("grade*_*.png"))]
    (WEB / "data" / "examples.json").write_text(json.dumps(examples))

    # ---- Parity fixture: Python outputs for every validation X-ray ----
    meta = [r for r in csv.DictReader(open(PROJECT / "data/processed/metadata.csv")) if r["split"] == "val"]
    with np.load(PROJECT / "data/processed/val.npz") as d:
        X_val, y_val = d["X"], d["y"]
    probs = np.mean([m.predict(X_val[..., None].astype("float32") / 255, batch_size=64, verbose=0)
                     for m in models], axis=0)
    items = []
    for i, row in enumerate(meta):
        src = Path(row["path"])
        rel = f"val/{row['grade']}/{src.name}"
        (WEB / "parity" / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, WEB / "parity" / rel)
        check = pp.load_and_preprocess(src)
        assert np.array_equal(check, X_val[i]), f"metadata order mismatch at {rel}"
        items.append({"path": rel, "label": int(y_val[i]), "hash": fnv1a(X_val[i].tobytes()),
                      "probs": [round(float(p), 6) for p in probs[i]]})
    (WEB / "parity" / "expected.json").write_text(json.dumps(items))
    prior = np.array(TRAIN_COUNTS) / sum(TRAIN_COUNTS)
    pred = (probs / prior).argmax(1)
    bal = np.mean([(pred[y_val == g] == g).mean() for g in range(5)])
    print(f"Parity fixture: {len(items)} validation X-rays, Python balanced accuracy {bal:.4f}")
    print("Done. Serve the site with:  cd web && python -m http.server 8000")


if __name__ == "__main__":
    main()
