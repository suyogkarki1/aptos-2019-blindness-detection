"""Streamlit UI: upload a retina image and get a diabetic retinopathy grade.

Run with:  streamlit run app.py
"""
import os

import cv2
import numpy as np
import streamlit as st
import torch

from src.grade_info import CAUSES, GRADE_INFO, PREVENTION, PROGRESSION, SYMPTOMS
from src.inference import check_fundus, load_model, predict

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "models", "best_model_effi.pth")
WEIGHTS_URL = ("https://github.com/suyogkarki1/aptos-2019-blindness-detection/"
               "releases/download/v1.0/best_model_effi.pth")
WEIGHTS_SHA256 = "f82d2fc37e96dcac4c3bfcc7e9ab9567485658ed94e4979cad1f88d66e02f74a"
EXAMPLES = os.path.join(HERE, "assets", "examples")

st.set_page_config(page_title="APTOS DR Grading", page_icon="👁️", layout="wide")

# Translucent backgrounds so the cards work with both light and dark themes.
# Streamlit's markdown treats indented lines as code, so HTML is kept flush-left.
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 0.8rem; margin-bottom: 1rem;
background: linear-gradient(135deg, rgba(183,28,28,0.14), rgba(239,108,0,0.10) 50%, rgba(46,125,50,0.10));
border: 1px solid rgba(128,128,128,0.25);}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin: 0.3rem 0 0; opacity: 0.8;}
.card {padding: 1rem 1.1rem; border-radius: 0.7rem; height: 100%;
background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.22);}
.card h4 {margin: 0 0 0.35rem; font-size: 1rem;}
.card p {margin: 0; font-size: 0.9rem; opacity: 0.85;}
.grade-card {padding: 1.1rem 1.2rem; border-radius: 0.7rem; color: white;}
.grade-card .lbl {font-size: 0.85rem; opacity: 0.85;}
.grade-card .val {font-size: 2rem; font-weight: 700; line-height: 1.2;}
.grade-card .sub {font-size: 0.95rem; opacity: 0.92; margin-top: 0.2rem;}
.badge {display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
font-size: 0.75rem; font-weight: 600; background: rgba(255,255,255,0.25); margin-top: 0.5rem;}
.scale {position: relative; margin: 1.4rem 0 0.4rem;}
.scale .bar {display: flex; height: 12px; border-radius: 6px; overflow: hidden;}
.scale .bar div {flex: 1;}
.scale .marker {position: absolute; top: -14px; transform: translateX(-50%); font-size: 0.9rem;}
.scale .ticks {display: flex; font-size: 0.72rem; opacity: 0.75; margin-top: 0.25rem;}
.scale .ticks span {flex: 1; text-align: center;}
.chip {display: inline-block; width: 0.75rem; height: 0.75rem; border-radius: 3px;
margin-right: 0.45rem; vertical-align: middle;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(WEIGHTS, device), device


def example_path(grade):
    p = os.path.join(EXAMPLES, f"grade_{grade}.jpg")
    return p if os.path.exists(p) else None


def severity_scale(score):
    # each grade owns a 20% segment; score 0 sits in the middle of the first one
    pos = float(np.clip((score + 0.5) / 5, 0, 1)) * 100
    segs = "".join(f"<div style='background:{GRADE_INFO[g]['color']}'></div>" for g in range(5))
    ticks = "".join(f"<span>{g}</span>" for g in range(5))
    return (f"<div class='scale'><div class='marker' style='left:{pos:.1f}%'>▼</div>"
            f"<div class='bar'>{segs}</div><div class='ticks'>{ticks}</div></div>")


def card(title, body):
    return f"<div class='card'><h4>{title}</h4><p>{body}</p></div>"


# ---------- sidebar ----------
st.sidebar.header("Model")
if not os.path.exists(WEIGHTS):
    # first run after cloning: fetch the weights from the GitHub release
    try:
        with st.spinner("Downloading model weights (43 MB, first run only)..."):
            os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
            torch.hub.download_url_to_file(WEIGHTS_URL, WEIGHTS, hash_prefix=WEIGHTS_SHA256)
    except Exception as e:
        st.sidebar.error(
            f"Could not download the model ({e}). Download `best_model_effi.pth` "
            f"from {WEIGHTS_URL} into the `models` folder and refresh."
        )
        st.stop()

model, device = get_model()
st.sidebar.success(f"EfficientNet-B3 (regression)\n\nDevice: {device.type.upper()}")

st.sidebar.subheader("Severity scale")
st.sidebar.markdown(
    "<br>".join(
        f"<span class='chip' style='background:{i['color']}'></span><b>{g}</b> · {i['name']}"
        for g, i in GRADE_INFO.items()
    ),
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Research/educational model trained on APTOS 2019 (+ 2015 EyePACS data). "
    "**Not a medical device.** Do not use it for diagnosis."
)

# ---------- header ----------
st.markdown(
    "<div class='hero'><h1>👁️ Diabetic Retinopathy Grading</h1>"
    "<p>Upload a fundus (retina) photograph to estimate diabetic retinopathy "
    "severity on the 0–4 clinical scale, and learn what each stage means.</p></div>",
    unsafe_allow_html=True,
)

tab_analyse, tab_learn, tab_model = st.tabs(
    ["🔍 Analyse image", "📖 Understanding DR", "🧠 About the model"]
)

# ---------- tab 1: analyse ----------
with tab_analyse:
    # changing the uploader's key is the only way to empty it programmatically
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    files = st.file_uploader(
        "Retina image(s) — select several at once with Ctrl/Shift, or drop more in later",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if files and st.button("🗑️ Clear images and try another", type="primary"):
        st.session_state.uploader_key += 1
        st.rerun()

    if not files:
        st.info(
            "Upload a **fundus photograph**: the round, orange image of the back of "
            "the eye taken with a retinal camera. Ordinary photos of the eye will "
            "not give meaningful results. See the *Understanding DR* tab for examples."
        )

    for file in files or []:
        data = np.frombuffer(file.getvalue(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error(f"{file.name}: could not read image")
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        st.subheader(file.name)
        is_fundus, reasons = check_fundus(img)
        if not is_fundus:
            w1, w2 = st.columns([1, 2.2])
            w1.image(img, caption="Uploaded image", use_container_width=True)
            with w2:
                st.warning(
                    "**This doesn't look like a fundus (retina) photograph.**\n\n"
                    + "\n".join(f"- {r}" for r in reasons)
                    + "\n\nThe model only understands images taken with a retinal "
                    "camera, so its grade for this image would be meaningless."
                )
                ex = example_path(2)
                if ex:
                    st.image(ex, width=220, caption="What a fundus photo looks like")
                force = st.checkbox(
                    "It is a fundus photo — analyse anyway",
                    key=f"force_{st.session_state.uploader_key}_{file.file_id}",
                )
            if not force:
                st.markdown("---")
                continue

        with st.spinner(f"Analysing {file.name}..."):
            res = predict(model, img, device)

        grade = res["grade"]
        info = GRADE_INFO[grade]

        if not is_fundus:
            st.caption("⚠️ Analysed despite failing the fundus check. Treat this result with caution.")
        c1, c2, c3 = st.columns([1, 1, 1.2])
        c1.image(img, caption="Original", use_container_width=True)
        c2.image(res["processed"], caption="Preprocessed (what the model sees)",
                 use_container_width=True)
        with c3:
            st.markdown(
                f"<div class='grade-card' style='background:{info['color']}'>"
                f"<div class='lbl'>Predicted grade</div>"
                f"<div class='val'>{grade} · {info['name']}</div>"
                f"<div class='sub'>{info['short']}</div>"
                f"<span class='badge'>{info['urgency']}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(severity_scale(res["score"]), unsafe_allow_html=True)
            st.caption(
                f"Model score **{res['score']:.2f}** on a continuous 0–4 scale "
                "(cut-offs at 0.5 / 1.5 / 2.5 / 3.5)."
            )

        d1, d2 = st.columns([2, 1])
        with d1:
            st.markdown("##### What this means")
            st.write(info["meaning"])
            st.markdown("##### Signs an eye doctor looks for at this stage")
            st.markdown("\n".join(f"- {s}" for s in info["signs"]))
            st.markdown("##### Typical next step")
            st.write(info["action"])
        with d2:
            ex = example_path(grade)
            if ex:
                st.markdown(f"##### Reference: a typical grade {grade}")
                st.image(ex, use_container_width=True,
                         caption=f"Labelled APTOS training image, grade {grade}")
        st.markdown("---")

# ---------- tab 2: education ----------
with tab_learn:
    st.markdown("### What is diabetic retinopathy?")
    st.write(
        "Diabetic retinopathy (DR) is damage to the retina, the light-sensitive layer "
        "at the back of the eye, caused by diabetes. It is one of the leading causes "
        "of preventable blindness in working-age adults. It develops silently: "
        "by the time vision is affected the disease is often advanced, but when "
        "caught early through screening, treatment can prevent most vision loss."
    )

    st.markdown("### How it develops")
    cols = st.columns(len(PROGRESSION))
    for col, (title, body) in zip(cols, PROGRESSION):
        col.markdown(card(title, body), unsafe_allow_html=True)

    st.markdown("### Causes and risk factors")
    for row in range(0, len(CAUSES), 3):
        cols = st.columns(3)
        for col, (title, body) in zip(cols, CAUSES[row:row + 3]):
            col.markdown(card(title, body), unsafe_allow_html=True)
        st.write("")

    st.markdown("### The five stages")
    st.caption("Example images are labelled fundus photographs from the APTOS 2019 training set.")
    for g, info in GRADE_INFO.items():
        ci, ct = st.columns([1, 2])
        ex = example_path(g)
        if ex:
            ci.image(ex, use_container_width=True)
        with ct:
            st.markdown(
                f"<h4 style='margin-bottom:0.2rem'><span class='chip' "
                f"style='background:{info['color']}'></span>Grade {g} · {info['name']}</h4>",
                unsafe_allow_html=True,
            )
            st.write(info["meaning"])
            st.markdown("**Signs:** " + "; ".join(info["signs"]) + ".")
            st.markdown(f"**Typical next step:** {info['action']}")
        st.write("")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("### Symptoms")
        st.markdown("\n".join(f"- {s}" for s in SYMPTOMS))
    with s2:
        st.markdown("### Prevention")
        st.markdown("\n".join(f"- {s}" for s in PREVENTION))

    st.caption(
        "Stages follow the International Clinical Diabetic Retinopathy severity scale. "
        "Follow-up intervals are general guidance only; an eye care professional "
        "decides the actual plan."
    )

# ---------- tab 3: model ----------
with tab_model:
    m1, m2, m3 = st.columns(3)
    m1.metric("Kaggle private QWK", "0.894")
    m2.metric("Local test QWK", "0.915")
    m3.metric("Local test accuracy", "0.80")

    st.markdown("### How a prediction is made")
    steps = [
        ("1. Crop", "The black border around the eye is removed so only the retina remains."),
        ("2. Enhance", "Ben Graham method: resize to 320×320 and subtract a Gaussian blur to highlight lesions."),
        ("3. Predict", "EfficientNet-B3 outputs one continuous severity score, averaged over the image and its mirror (TTA)."),
        ("4. Grade", "The score is rounded to a grade with cut-offs at 0.5, 1.5, 2.5 and 3.5."),
    ]
    cols = st.columns(4)
    for col, (title, body) in zip(cols, steps):
        col.markdown(card(title, body), unsafe_allow_html=True)

    st.markdown("### Training")
    st.markdown(
        "- **Data:** APTOS 2019 (3,662 images) plus ~17k images from the 2015 "
        "Diabetic Retinopathy Detection competition\n"
        "- **Model:** EfficientNet-B3, ImageNet pretrained, single regression output\n"
        "- **Loss:** MSE, so near-misses are penalised less than big errors, matching "
        "the Quadratic Weighted Kappa metric\n"
        "- **Augmentation:** flips, full rotation, brightness/contrast jitter"
    )
    st.markdown("### Limitations")
    st.markdown(
        "- Trained only on fundus photographs; other images give meaningless grades\n"
        "- Image quality, camera type and population differences affect accuracy\n"
        "- Detects DR severity only, not other eye diseases such as glaucoma or macular degeneration\n"
        "- A screening aid for research and education, not a diagnosis"
    )
