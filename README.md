# APTOS 2019 Blindness Detection

Grading diabetic retinopathy severity (0–4) from retina images, built in PyTorch and trained on free Kaggle GPUs.

**Competition:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
**Metric:** Quadratic Weighted Kappa (QWK)

| Grade | Meaning |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Final result

**Private leaderboard QWK: 0.894** (public: 0.781), with a single EfficientNet-B3 model.
For reference, the competition winner scored 0.936 private.

## Approach

- **Preprocessing:** Ben Graham method (circle crop + Gaussian-blur contrast enhancement), resized to 320×320
- **Model:** EfficientNet-B3 (ImageNet pretrained) with a single regression output
- **Loss:** MSE (regression), predictions converted to grades with thresholds
- **External data:** ~17k images from the 2015 Diabetic Retinopathy Detection competition added to training (all diseased images + 8,000 healthy); validation and test kept APTOS-only
- **Augmentation:** horizontal/vertical flips, full rotation, brightness/contrast jitter
- **Training:** Adam, cosine LR schedule, mixed precision, best checkpoint chosen by validation QWK
- **Inference:** flip test-time augmentation (TTA), simple rounding thresholds

## Results

| Kaggle submission | Model | Private | Public |
|---|---|---|---|
| Baseline | ResNet50, classification | 0.848 | 0.658 |
| Version 1 | ResNet50, regression + tuned thresholds | 0.846 | 0.618 |
| Version 2 | ResNet50, regression + rounding | 0.849 | 0.624 |
| **Version 3** | **EfficientNet-B3 + 2015 external data** | **0.894** | **0.781** |

Local results for the final model (held-out APTOS test split): **QWK 0.915**, accuracy 0.80.

## Key lessons

1. **Distribution shift was the main bottleneck.** With ResNet50, local test QWK was ~0.89, but the private leaderboard stayed around 0.85. The hidden test images come from a different distribution than the training set.
2. **Post-processing could not fix it.** Thresholds tuned on validation scored slightly *worse* on the leaderboard than simple rounding (0.846 vs 0.849), because they fitted the training distribution.
3. **Diverse external data did.** Adding ~17k images from the 2015 competition, together with EfficientNet-B3, raised the private score by **+0.045** (0.849 → 0.894).
4. **Regression suits ordinal grades.** Predicting a single continuous score rewards near-misses, which matches how QWK scores predictions.

## Repository structure

```
aptos/
├── app.py                  # Streamlit web UI (entry point)
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_training.ipynb   # EfficientNet-B3 training (with 2015 data), threshold tuning, evaluation
│   └── 02_submission.ipynb # offline Kaggle submission for the EfficientNet-B3 model (Version 3)
├── src/
│   ├── inference.py        # preprocessing, fundus image check, model loading, flip-TTA prediction
│   └── grade_info.py       # educational text for each DR grade, causes, symptoms
├── models/                 # best_model_effi.pth, auto-downloaded from the v1.0 release (not tracked by git)
└── assets/examples/        # reference image per grade from APTOS train set (not tracked by git)
```

## Training on Kaggle

1. Open the notebooks in `notebooks/` on Kaggle (GPU enabled).
2. **Training notebook:** attach the APTOS 2019 competition data and a resized 2015 Diabetic Retinopathy dataset; internet on (for pretrained weights).
3. **Submission notebook:** attach the competition data and the training notebook's output (`best_model_effi.pth`); internet off.

## Web app

A Streamlit app for grading your own fundus images with the trained model.

### Setup

1. `pip install -r requirements.txt`
2. `streamlit run app.py`, then open http://localhost:8501

On first run the app downloads the trained weights (`best_model_effi.pth`, 43 MB) from the [v1.0 release](https://github.com/suyogkarki1/aptos-2019-blindness-detection/releases/tag/v1.0) into `models/` and verifies the checksum. To do it manually, download the file from the release into `models/`.

Optional: put one labelled APTOS training image per grade in `assets/examples/` as `grade_0.jpg` … `grade_4.jpg` to show reference images in the app.

### Features

- **Analyse image:** upload one or more fundus photos and get the predicted grade (0–4), the continuous model score on a colour-coded severity bar, what that stage means, the signs an eye doctor looks for, the typical next step, and a reference image of the same grade
- **Fundus image check:** before grading, each image is checked for the traits of a retina photo (a large, round, orange-red disc on a black background). Other images (selfies, webcam shots, screenshots) get a warning instead of a meaningless grade, with an "analyse anyway" override. On held-out images it accepted 99.7% of APTOS fundus photos and rejected 100% of ~500 everyday photos and screenshots
- **Understanding DR:** what diabetic retinopathy is, how it develops, causes and risk factors, all five stages with example images, symptoms and prevention
- **About the model:** scores, the prediction pipeline and limitations

For research and education only, not for medical diagnosis.

## Possible improvements

- Ensemble several models (different seeds or input sizes)
- Use more of the 2015 dataset
- Higher input resolution (384–512)
- Generalized mean pooling and other techniques used by top solutions
- Camera capture in the web app for smartphone fundus adapters

## Tech stack

Python · PyTorch · torchvision · OpenCV · scikit-learn · pandas · NumPy · Streamlit · Kaggle
