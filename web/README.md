---
title: KneeGrade
emoji: 🦴
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
license: cc-by-4.0
short_description: Knee OA KL grading in your browser, with Grad-CAM
---

# KneeGrade

Grades knee osteoarthritis severity on the Kellgren–Lawrence scale (0–4) from a front-view knee X-ray, and shows a Grad-CAM heatmap of where the model looked.

**Everything runs in your browser** with TensorFlow.js. The two custom CNNs (trained from scratch) download once; X-rays are never uploaded anywhere. The browser pipeline reproduces the original Python pipeline exactly: identical preprocessing, matching probabilities, and the same validation balanced accuracy.

- Test balanced accuracy 0.644 (95% CI 0.621–0.666), quadratic weighted kappa 0.753, on 1,656 held-out X-rays.
- Model card: [afzal2003/knee-oa-kl-grading](https://huggingface.co/afzal2003/knee-oa-kl-grading)
- Analysis history is stored only in your own browser.

**Research and education demo only. Not a medical device and not for diagnosis.**

Data: Chen, Pingjun (2018), *Knee Osteoarthritis Severity Grading Dataset*, Mendeley Data, DOI 10.17632/56rmx5bjcr.1, CC BY 4.0. Organized from the Osteoarthritis Initiative.
