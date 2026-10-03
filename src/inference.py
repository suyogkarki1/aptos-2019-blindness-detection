"""Model loading and single-image inference, matching 02_submission.ipynb."""
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 320
BEST_T = [0.5, 1.5, 2.5, 3.5]

GRADES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def crop_circle(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    ys, xs = np.where(mask)
    return img[ys.min():ys.max()+1, xs.min():xs.max()+1]


def ben_graham(img, size=IMG_SIZE, sigma=10):
    """img: RGB uint8 array. Returns the preprocessed PIL image."""
    img = cv2.resize(crop_circle(img), (size, size))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)
    return Image.fromarray(img)


def check_fundus(img):
    """Heuristic check that img (RGB uint8) looks like a fundus photograph:
    a bright, round, centred retina disc on a black background, in colour.

    Thresholds were tuned on APTOS train/test images vs. ~1,000 everyday
    photos and screenshots (held-out: 99.7% of fundus images accepted,
    100% of other images rejected). Returns (ok, list of failure reasons).
    """
    h, w = img.shape[:2]
    s = 512 / max(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # fundus photos have black corners outside the circular retina
    k = max(4, int(min(h, w) * 0.08))
    corners = np.concatenate([gray[:k, :k].ravel(), gray[:k, -k:].ravel(),
                              gray[-k:, :k].ravel(), gray[-k:, -k:].ravel()])
    dark_corners = (corners < 30).mean()

    n, labels, stats, _ = cv2.connectedComponentsWithStats((gray > 15).astype(np.uint8))
    if n < 2:
        return False, ["The image is almost completely black."]
    disc = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    fg = disc.mean()

    # how well the bright region fills its enclosing circle (clipped to the frame,
    # so circles with the top/bottom cut off still count)
    cnts, _ = cv2.findContours(disc.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    circle = np.zeros_like(gray)
    cv2.circle(circle, (int(cx), int(cy)), int(r), 1, -1)
    roundness = disc.sum() / max(circle.sum(), 1)
    size = r / (min(h, w) / 2)

    rgb = img[disc].astype(float).mean(0)
    red = rgb[0] / (rgb.sum() + 1e-6)
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[disc][:, 1].mean() / 255

    reasons = []
    if dark_corners < 0.4 or fg > 0.995:
        reasons.append("No dark background around the eye. Fundus photos show the "
                       "retina as a bright disc surrounded by black.")
    if fg < 0.3 or size < 0.85:
        reasons.append("The bright region is too small to be a retina photo.")
    if roundness < 0.9:
        reasons.append("The bright region is not round like a retina image.")
    # reddish-orange retina, or a tinted camera with a clean circle and real colour
    if not ((red >= 0.36 and sat >= 0.12) or (red >= 0.30 and sat >= 0.18 and roundness >= 0.95)):
        reasons.append("The colours don't match a retina (expected orange-red tones, "
                       "not grey or other colours).")
    return not reasons, reasons


def load_model(path, device):
    """Loads the final EfficientNet-B3 regression model (best_model_effi.pth)."""
    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 1),
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, img_rgb, device):
    """Runs flip TTA on one RGB uint8 image.

    Returns dict with grade, score (continuous severity) and the model input.
    """
    processed = ben_graham(img_rgb)
    x = transform(processed).unsqueeze(0).to(device)
    out = (model(x) + model(torch.flip(x, dims=[3]))) / 2

    score = out.squeeze().item()
    grade = int(np.digitize(score, BEST_T))
    return {"grade": grade, "score": score, "processed": processed}
