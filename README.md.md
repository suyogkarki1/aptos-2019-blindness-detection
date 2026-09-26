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
aptos-2019-blindness-detection/
├── README.md
├── requirements.txt
├── 01_training.ipynb      # EfficientNet-B3 training (with 2015 data), threshold tuning, evaluation
└── 02_submission.ipynb    # offline Kaggle submission for the EfficientNet-B3 model (Version 3)
```

## How to run

1. Open the notebooks on Kaggle (GPU enabled).
2. **Training notebook:** attach the APTOS 2019 competition data and a resized 2015 Diabetic Retinopathy dataset; internet on (for pretrained weights).
3. **Submission notebook:** attach the competition data and the training notebook's output (`best_model_effi.pth`); internet off.

## Possible improvements

- Ensemble several models (different seeds or input sizes)
- Use more of the 2015 dataset
- Higher input resolution (384–512)
- Generalized mean pooling and other techniques used by top solutions

## Tech stack

Python · PyTorch · torchvision · OpenCV · scikit-learn · pandas · NumPy · Kaggle
