---
title: KneeGrade
emoji: 🦴
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: cc-by-4.0
short_description: Knee osteoarthritis KL grading from X-rays, with Grad-CAM
---

# KneeGrade

Grades knee osteoarthritis severity on the Kellgren–Lawrence scale (0–4) from a front-view knee X-ray, using two custom CNNs trained from scratch, and shows a Grad-CAM heatmap of where the model looked.

- Test balanced accuracy 0.644 (95% CI 0.621–0.666), quadratic weighted kappa 0.753, on 1,656 held-out X-rays.
- Models: [afzal2003/knee-oa-kl-grading](https://huggingface.co/afzal2003/knee-oa-kl-grading)
- Uploaded X-rays are processed in memory and not stored. Analysis history stays in your own browser.

**Research and education demo only. Not a medical device and not for diagnosis.**

Data: Chen, Pingjun (2018), *Knee Osteoarthritis Severity Grading Dataset*, Mendeley Data, DOI 10.17632/56rmx5bjcr.1, CC BY 4.0. Organized from the Osteoarthritis Initiative.
