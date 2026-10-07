"""
Parkinson's Disease Multimodal Predictor — Streamlit App
==========================================================
Wraps the models trained in:
  - parkinson_prediction_enhanced.ipynb   (voice features -> RandomForest)
  - handwriting_analysis_enhanced.ipynb   (spiral & wave drawings -> CNNs)

Combines the three signals with the same weighted average used in the
notebook's `predict_parkinsons_multimodal()` function:
  combined_score = 0.50 * voice_prob + 0.30 * spiral_prob + 0.20 * wave_prob

Run with:
    streamlit run app.py

Expected model files in the same folder (produced by the notebooks):
    parkinsons_model_balanced.pkl   (falls back to parkinsons_model.pkl)
    spiral_model.keras              (optional — see README)
    wave_model.keras                (optional — see README)
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Parkinson's Multimodal Predictor",
    page_icon="🧠",
    layout="wide",
)

# ----------------------------------------------------------------------
# Constants — must match the notebooks exactly
# ----------------------------------------------------------------------
VOICE_FEATURE_NAMES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE"
]

# (min, mean, max) from the training data — used to build sensible
# input widgets with a healthy-sample default pre-filled.
VOICE_FEATURE_RANGES = {
    "MDVP:Fo(Hz)":       (88.333, 154.229, 260.105),
    "MDVP:Fhi(Hz)":      (102.145, 197.105, 592.030),
    "MDVP:Flo(Hz)":      (65.476, 116.325, 239.170),
    "MDVP:Jitter(%)":    (0.00168, 0.00622, 0.03316),
    "MDVP:Jitter(Abs)":  (0.000007, 0.000044, 0.000260),
    "MDVP:RAP":          (0.00068, 0.00331, 0.02144),
    "MDVP:PPQ":          (0.00092, 0.00345, 0.01958),
    "Jitter:DDP":        (0.00204, 0.00992, 0.06433),
    "MDVP:Shimmer":      (0.00954, 0.02971, 0.11908),
    "MDVP:Shimmer(dB)":  (0.085, 0.282, 1.302),
    "Shimmer:APQ3":      (0.00455, 0.01622, 0.05647),
    "Shimmer:APQ5":      (0.00570, 0.01788, 0.07940),
    "MDVP:APQ":          (0.00719, 0.02414, 0.13778),
    "Shimmer:DDA":       (0.01364, 0.04699, 0.16942),
    "NHR":               (0.00065, 0.02485, 0.31482),
    "HNR":               (8.441, 21.886, 33.047),
    "RPDE":              (0.25657, 0.49854, 0.68515),
    "DFA":               (0.57428, 0.71810, 0.82529),
    "spread1":           (-7.96498, -5.68440, -2.43403),
    "spread2":           (0.00627, 0.22651, 0.45049),
    "D2":                (1.42329, 2.38183, 3.67116),
    "PPE":               (0.04454, 0.20655, 0.52737),
}

FUSION_WEIGHTS = (0.50, 0.30, 0.20)  # voice, spiral, wave
DEFAULT_THRESHOLD = 0.5

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# Model loading (cached so the app doesn't reload on every interaction)
# ----------------------------------------------------------------------
@st.cache_resource
def load_voice_model():
    balanced_path = os.path.join(MODEL_DIR, "parkinsons_model_balanced.pkl")
    legacy_path = os.path.join(MODEL_DIR, "parkinsons_model.pkl")
    if os.path.exists(balanced_path):
        return joblib.load(balanced_path), "parkinsons_model_balanced.pkl"
    if os.path.exists(legacy_path):
        return joblib.load(legacy_path), "parkinsons_model.pkl"
    return None, None


@st.cache_resource
def load_cnn_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(path)


voice_model, voice_model_name = load_voice_model()
spiral_model = load_cnn_model("spiral_model.keras")
wave_model = load_cnn_model("wave_model.keras")


# ----------------------------------------------------------------------
# Prediction helpers (mirrors the notebook functions)
# ----------------------------------------------------------------------
def get_voice_probability(feature_values: dict) -> float:
    row = [feature_values[f] for f in VOICE_FEATURE_NAMES]
    voice_df = pd.DataFrame([row], columns=VOICE_FEATURE_NAMES)
    return float(voice_model.predict_proba(voice_df)[0][1])


def get_spiral_probability(image: Image.Image) -> float:
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return float(spiral_model.predict(arr, verbose=0)[0][0])


def get_wave_probability(image: Image.Image) -> float:
    img = image.convert("RGB").resize((128, 128))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return float(wave_model.predict(arr, verbose=0)[0][0])


def verdict(prob: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    return "Parkinson's" if prob >= threshold else "Healthy"


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Model status")
    st.write(f"🎙️ Voice model: {'✅ ' + voice_model_name if voice_model else '❌ not found'}")
    st.write(f"🌀 Spiral CNN: {'✅ loaded' if spiral_model else '⚠️ not found (optional)'}")
    st.write(f"〰️ Wave CNN: {'✅ loaded' if wave_model else '⚠️ not found (optional)'}")
    st.markdown("---")
    st.caption(
        "Place `parkinsons_model_balanced.pkl`, `spiral_model.keras`, and "
        "`wave_model.keras` in the same folder as this app. See README.md "
        "for how to export them from the notebooks."
    )
    st.markdown("---")
    threshold = st.slider("Decision threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.05)

st.title("🧠 Parkinson's Disease Multimodal Predictor")
st.caption(
    "Combines a voice-recording model, a spiral-drawing model, and a "
    "wave-drawing model into one weighted prediction — same pipeline as "
    "the training notebooks. **For educational/demo purposes only — not "
    "a medical diagnostic tool.**"
)

tab_voice, tab_drawings, tab_combined = st.tabs(
    ["🎙️ Voice Analysis", "🌀 Drawing Analysis", "🧩 Combined Result"]
)

# ----------------------------------------------------------------------
# Tab 1 — Voice
# ----------------------------------------------------------------------
with tab_voice:
    st.subheader("Voice measurement features")
    st.caption(
        "Enter the 22 acoustic features extracted from a sustained vowel "
        "recording (as in the Parkinson's voice dataset)."
    )

    if voice_model is None:
        st.error(
            "No voice model found. Run the voice notebook and make sure "
            "`parkinsons_model_balanced.pkl` (or `parkinsons_model.pkl`) "
            "ends up in this app's folder."
        )
    else:
        with st.form("voice_form"):
            cols = st.columns(3)
            voice_inputs = {}
            for i, feat in enumerate(VOICE_FEATURE_NAMES):
                lo, mean, hi = VOICE_FEATURE_RANGES[feat]
                step = (hi - lo) / 200 if hi > lo else 0.01
                with cols[i % 3]:
                    voice_inputs[feat] = st.number_input(
                        feat, value=float(mean), format="%.5f", step=float(step),
                        key=f"voice_{feat}"
                    )
            submitted = st.form_submit_button("Predict from voice")

        if submitted:
            prob = get_voice_probability(voice_inputs)
            st.session_state["voice_prob"] = prob
            result = verdict(prob, threshold)
            c1, c2 = st.columns(2)
            c1.metric("Parkinson's probability", f"{prob:.1%}")
            c2.metric("Prediction", result)
            st.progress(min(max(prob, 0.0), 1.0))

# ----------------------------------------------------------------------
# Tab 2 — Drawings
# ----------------------------------------------------------------------
with tab_drawings:
    st.subheader("Spiral & wave drawing images")
    st.caption(
        "Upload a hand-drawn spiral and/or wave (e.g. scanned from the "
        "standard Parkinson's drawing test)."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Spiral drawing**")
        spiral_file = st.file_uploader(
            "Upload spiral image", type=["png", "jpg", "jpeg"], key="spiral_upload"
        )
        if spiral_file is not None:
            img = Image.open(spiral_file)
            st.image(img, caption="Uploaded spiral", width=250)
            if spiral_model is None:
                st.warning("Spiral CNN model not found — skipping this signal.")
            elif st.button("Predict from spiral"):
                prob = get_spiral_probability(img)
                st.session_state["spiral_prob"] = prob
                st.metric("Parkinson's probability", f"{prob:.1%}")
                st.metric("Prediction", verdict(prob, threshold))

    with col2:
        st.markdown("**Wave drawing**")
        wave_file = st.file_uploader(
            "Upload wave image", type=["png", "jpg", "jpeg"], key="wave_upload"
        )
        if wave_file is not None:
            img = Image.open(wave_file)
            st.image(img, caption="Uploaded wave", width=250)
            if wave_model is None:
                st.warning("Wave CNN model not found — skipping this signal.")
            elif st.button("Predict from wave"):
                prob = get_wave_probability(img)
                st.session_state["wave_prob"] = prob
                st.metric("Parkinson's probability", f"{prob:.1%}")
                st.metric("Prediction", verdict(prob, threshold))

# ----------------------------------------------------------------------
# Tab 3 — Combined
# ----------------------------------------------------------------------
with tab_combined:
    st.subheader("Weighted multimodal result")
    w_voice, w_spiral, w_wave = FUSION_WEIGHTS
    st.caption(
        f"combined_score = {w_voice:.2f} × voice + {w_spiral:.2f} × spiral + "
        f"{w_wave:.2f} × wave  (only available signals are used, "
        f"reweighted proportionally)"
    )

    available = {
        "voice": (st.session_state.get("voice_prob"), w_voice),
        "spiral": (st.session_state.get("spiral_prob"), w_spiral),
        "wave": (st.session_state.get("wave_prob"), w_wave),
    }
    present = {k: v for k, (v, w) in available.items() if v is not None}

    if not present:
        st.info(
            "Run at least one prediction in the **Voice Analysis** or "
            "**Drawing Analysis** tabs first — results appear here "
            "automatically."
        )
    else:
        weight_sum = sum(available[k][1] for k in present)
        combined = sum(available[k][0] * available[k][1] for k in present) / weight_sum

        cols = st.columns(len(present) + 1)
        for i, k in enumerate(present):
            cols[i].metric(f"{k.capitalize()} probability", f"{present[k]:.1%}")
        cols[-1].metric("Combined score", f"{combined:.1%}")

        result = verdict(combined, threshold)
        if result == "Parkinson's":
            st.error(f"### Prediction: {result}")
        else:
            st.success(f"### Prediction: {result}")
        st.progress(min(max(combined, 0.0), 1.0))

        if len(present) < 3:
            missing = [k for k in available if k not in present]
            st.caption(f"Missing signal(s): {', '.join(missing)} — weights were renormalized over the available signals.")