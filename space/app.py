"""KneeGrade web app: a FastAPI backend serving the frozen pipeline, plus a static frontend.

Models are downloaded from the Hugging Face Hub. The server stores nothing:
each user's analysis history lives in their own browser (IndexedDB).

Run locally:  python app.py   then open http://127.0.0.1:7860
"""
import base64
import csv
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import keras
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download

import gradcam as gc
import preprocessing as pp

REPO_ID = os.environ.get("MODEL_REPO", "afzal2003/knee-oa-kl-grading")
HERE = Path(__file__).resolve().parent
STATIC, ASSETS, EXAMPLES = HERE / "static", HERE / "assets", HERE / "examples"
MAX_BYTES = 10 * 1024 * 1024
JOINT_BAND = (slice(70, 154), slice(20, 204))          # same band used in the explainability analysis

# ---------- Load the frozen pipeline from the Hub (once, at startup) ----------
config = json.loads(Path(hf_hub_download(REPO_ID, "pipeline_config.json")).read_text())
MODELS = [keras.models.load_model(hf_hub_download(REPO_ID, f)) for f in config["members"]]
GRAD_MODELS = gc.build_gradcam_models(MODELS, layers=(config["gradcam_layer"],))
PRIOR = np.array(config["train_counts_by_grade"], dtype=float)
PRIOR /= PRIOR.sum()

app = FastAPI(title="KneeGrade", docs_url=None, redoc_url=None)


def read_xray(upload: UploadFile):
    """Decode an uploaded image, resize to 224x224 grayscale, run the preprocessing pipeline."""
    data = upload.file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(400, "The file is empty. Choose a PNG or JPG image.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "This file is larger than 10 MB. Upload a smaller image.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(415, "This file isn't a readable image. Upload a PNG or JPG.")
    if img.shape != (224, 224):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return pp.preprocess(img), bool(pp.polarity_score(img) < 0)


def data_url(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "The result image couldn't be encoded.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@app.get("/api/meta")
def meta():
    return {"model_repo": REPO_ID, "version": config.get("version", "1.0"),
            "grade_names": config["grade_names"], "train_counts": config["train_counts_by_grade"]}


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    x_pre, inverted = read_xray(file)
    x = x_pre[None, ..., None].astype("float32") / 255.0
    probs = np.mean([m(x, training=False).numpy()[0] for m in MODELS], axis=0)
    scores = probs / PRIOR
    scores /= scores.sum()
    return {
        "grade": int(scores.argmax()),
        "top_grade": int(probs.argmax()),
        "probabilities": [round(float(p), 4) for p in probs],
        "adjusted_scores": [round(float(s), 4) for s in scores],
        "inverted_corrected": inverted,
        "preprocessed": data_url(x_pre),
        "elapsed_ms": round((time.perf_counter() - start) * 1000),
    }


@app.post("/api/gradcam")
def gradcam(file: UploadFile = File(...), grade: int = Form(...)):
    if not 0 <= grade <= 4:
        raise HTTPException(422, "Grade must be between 0 and 4.")
    x_pre, _ = read_xray(file)
    cam = gc.ensemble_gradcam(GRAD_MODELS, x_pre, grade)
    colour = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)      # BGR
    alpha = np.uint8(np.clip(cam * 1.5, 0, 1) * 255)                          # weak signal fades out
    total = float(cam.sum())
    share = float(cam[JOINT_BAND].sum() / total) if total > 0 else None
    return {"heatmap": data_url(np.dstack([colour, alpha])), "joint_share": share}


@app.get("/api/training-curves")
def training_curves():
    curves = {}
    for run in ("exp1_augmentation", "exp3_aug_wider"):
        path = ASSETS / f"{run}_history.csv"
        if path.exists():
            with path.open() as f:
                rows = list(csv.DictReader(f))
            if rows:
                curves[run] = {key: [float(r[key]) for r in rows] for key in rows[0]}
    return curves


@app.get("/api/examples")
def examples():
    items = []
    for p in sorted(EXAMPLES.glob("*.png")):
        m = re.match(r"grade(\d)_", p.name)
        items.append({"url": f"/examples/{p.name}", "name": p.name,
                      "label": int(m.group(1)) if m else None})
    return items


app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
app.mount("/examples", StaticFiles(directory=EXAMPLES), name="examples")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 7860)))
