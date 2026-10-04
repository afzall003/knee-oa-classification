---
library_name: keras
pipeline_tag: image-classification
license: cc-by-4.0
tags:
  - medical-imaging
  - x-ray
  - knee-osteoarthritis
  - kellgren-lawrence
  - tensorflow
  - keras
  - grad-cam
---

# Knee osteoarthritis KL grading (custom CNN ensemble)

Two custom convolutional neural networks, designed and **trained from scratch** in TensorFlow/Keras, that grade knee osteoarthritis severity on the **Kellgren–Lawrence (KL) scale, 0–4**, from front-view knee X-rays cropped to the joint. No pretrained weights or transfer learning were used.

> **Not a medical device.** Research and portfolio project only. Not clinically validated; must not be used for diagnosis or treatment decisions.

- **Code, notebooks and full write-up:** https://github.com/afzall003/knee-oa-classification
- **Live demo:** https://huggingface.co/spaces/afzal2003/kneegrade

## Results on the held-out test set (1,656 X-rays)

| Metric | Value | 95% confidence interval |
|---|---|---|
| Balanced accuracy | **0.644** | 0.621 – 0.666 |
| Quadratic weighted kappa | **0.753** | 0.730 – 0.775 |
| Accuracy | 0.542 | |
| Macro F1 | 0.604 | |

| Grade | 0 Healthy | 1 Doubtful | 2 Minimal | 3 Moderate | 4 Severe |
|---|---|---|---|---|---|
| Recall | 0.44 | 0.44 | 0.58 | 0.80 | 0.96 |
| Precision | 0.80 | 0.24 | 0.59 | 0.71 | 0.69 |

Confidence intervals come from 2,000 bootstrap resamples. The pipeline was committed to version control before the test set was evaluated, and the test set was used exactly once. Almost all errors are to an adjacent grade; no healthy knee was ever predicted as grade 4.

**The main weakness is the grade 0 / 1 / 2 boundary.** With the decision rule below, the model recognizes grade 1 (recall 0.44) but flags 46% of healthy knees as grade 1. With a plain argmax rule, balanced accuracy is the same (0.646) but grade 1 is almost never predicted (recall 0.02) and healthy-knee recall rises to 0.82.

## Intended use

- Demonstrating and studying automated KL grading on research data.
- Education about deep learning for medical imaging, class imbalance and explainability.

**Out of scope:** clinical diagnosis or triage; images other than front-view knee radiographs cropped to the joint (the model will still output a grade for any image, but it is meaningless); populations, scanners or sites other than the training data without independent validation.

## Files

| File | Description |
|---|---|
| `exp1_augmentation.keras` | CNN, 4 blocks, filters 32-64-128-256, 1,175,845 parameters |
| `exp3_aug_wider.keras` | Same architecture, 1.5× wider (48-96-192-384) |
| `preprocessing.py` | Required image preprocessing (see below) |
| `pipeline_config.json` | Ensemble members, decision rule and training-set grade counts |

## How the pipeline works

1. **Preprocessing** (`preprocessing.py`): inverted X-rays are detected and flipped back; a 1st–99th percentile contrast stretch is measured on the central joint region; then CLAHE (clip limit 2.0, 8×8 tiles). Input must be a **224×224 grayscale** image; resize first if needed.
2. **Ensemble:** both models output softmax probabilities over grades 0–4, which are averaged.
3. **Prior-adjusted decision:** each probability is divided by that grade's frequency in the training set, and the highest score wins. This corrects the model's lean toward common grades and was selected on validation data only. The resulting scores are decision scores, not calibrated probabilities.

## Usage

```python
import importlib.util
import keras
import numpy as np
from huggingface_hub import hf_hub_download

repo = "afzal2003/knee-oa-kl-grading"
models = [keras.models.load_model(hf_hub_download(repo, f))
          for f in ["exp1_augmentation.keras", "exp3_aug_wider.keras"]]

spec = importlib.util.spec_from_file_location("preprocessing", hf_hub_download(repo, "preprocessing.py"))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)

img = pp.load_and_preprocess("knee_xray.png")            # 224x224 grayscale knee X-ray
x = img[None, ..., None].astype("float32") / 255.0
probs = np.mean([m.predict(x, verbose=0) for m in models], axis=0)[0]

train_counts = np.array([2286, 1046, 1516, 757, 173])
grade = int((probs / (train_counts / train_counts.sum())).argmax())
print("KL grade:", grade, "| model probabilities:", probs.round(3))
```

## Training data

[Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity), derived from the Osteoarthritis Initiative (OAI) via Pingjun Chen's release on Mendeley Data (University of Florida). Images are 224×224 grayscale crops of the knee joint. Provided splits were used: 5,778 training, 826 validation and 1,656 test images. Grade 4 makes up only 3% of the training data (13:1 imbalance against grade 0).

## Training procedure

- Adam (learning rate 1e-3, halved on validation-loss plateaus), batch size 32, sparse categorical cross-entropy.
- Checkpointing and early stopping driven by **validation balanced accuracy**, computed by a custom callback.
- Augmentation: horizontal flip, rotation ±10°, shift ±5%, zoom ±10%, brightness and contrast ±10%.
- Trained on a single NVIDIA RTX 4060 laptop GPU.

## Explainability

Grad-CAM (deepest convolutional layer, averaged over both models) was validated with independent tests on test images:

- **70%** of the Grad-CAM signal falls in the central joint band, which covers 31% of the image.
- **Deletion test:** erasing the regions ranked most important drops confidence far faster than erasing random regions (area under curve 0.154 for occlusion and 0.170 for Grad-CAM, vs 0.422 for random).
- **Model-randomization sanity check:** passed (Spearman similarity −0.03 between trained and untrained networks).
- On an X-ray containing metal surgical hardware, the explanations ignore the hardware and focus on the joint.

## Limitations

- **Possible patient-level leakage:** most patients contribute both knees and the public data has no patient identifiers, so the test results may be slightly optimistic.
- **Single data source (OAI).** Performance on other sites, scanners and populations is unknown.
- **Grade 1 ("doubtful") is not reliably separated from its neighbours**, even on training data, consistent with known inter-reader disagreement.
- **Explanations are validated but not verified** against expert lesion annotations, which the dataset does not include.
- One training run per configuration; run-to-run variation is roughly ±0.02 in balanced accuracy.

## License and attribution

**Models:** released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Training data:** Chen, Pingjun (2018). *Knee Osteoarthritis Severity Grading Dataset*. Mendeley Data, V1. DOI: [10.17632/56rmx5bjcr.1](https://doi.org/10.17632/56rmx5bjcr.1). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Data organized from the [Osteoarthritis Initiative (OAI)](https://oai.epi-ucsf.org/datarelease/). Accessed via the Kaggle upload by Shashwat Tiwari.

**Changes made:** the dataset is not redistributed here. For training, images were preprocessed (inverted-image correction, percentile contrast stretch, CLAHE) and augmented. Neither the dataset authors nor the OAI endorse this project.

## Acknowledgements

Methods: Kellgren–Lawrence grading, CLAHE, Grad-CAM (Selvaraju et al.), CORAL ordinal regression (Cao et al.), model-randomization sanity checks (Adebayo et al.).
