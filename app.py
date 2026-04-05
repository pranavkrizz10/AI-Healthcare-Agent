"""
app.py
======
Streamlit web UI for the AI Health Assistant.

Run with:
    streamlit run app.py
"""

import os
import json
import joblib
import numpy as np
import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="centered",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("models", "disease_classifier.pkl")
SYMPTOM_PATH = os.path.join("models", "symptom_list.json")

# ── Advice database ────────────────────────────────────────────────────────────
ADVICE_DB: dict[str, str] = {
    "common cold":        "Rest, stay hydrated, and take over-the-counter cold remedies. See a GP if symptoms worsen after 10 days.",
    "typhoid":            "Seek medical attention immediately. Typhoid requires antibiotic treatment prescribed by a doctor.",
    "dengue":             "Go to A&E or a doctor urgently. Stay hydrated and avoid aspirin/ibuprofen.",
    "malaria":            "See a doctor immediately for a blood test and prescription anti-malarials.",
    "pneumonia":          "Visit a doctor promptly — you may need antibiotics or hospital care.",
    "diabetes":           "Consult your GP for blood-glucose testing and long-term management advice.",
    "hypertension":       "Monitor your blood pressure and book a GP appointment. Reduce salt and stress.",
    "migraine":           "Rest in a dark quiet room. Over-the-counter analgesics may help. See a GP for recurring migraines.",
    "allergy":            "Avoid known triggers. Antihistamines may relieve symptoms. Consult a GP if severe.",
    "urinary tract infection": "Drink plenty of water. A GP can prescribe antibiotics if needed.",
    "gastroenteritis":    "Stay hydrated (oral rehydration salts help). Symptoms usually resolve within a week.",
    "tuberculosis":       "See a doctor urgently. TB requires a long course of prescribed antibiotics.",
    "hepatitis":          "See a doctor for blood tests. Avoid alcohol. Hepatitis may require antiviral treatment.",
    "heart attack":       "CALL 999 / 112 IMMEDIATELY. Chew aspirin (300 mg) if available and not allergic.",
    "default":            "Please consult a qualified healthcare professional for a proper diagnosis and treatment plan.",
}


def get_advice(disease: str) -> str:
    return ADVICE_DB.get(disease.lower(), ADVICE_DB["default"])


# ── Load model & extractor (cached) ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_extractor():
    try:
        from extractor import SymptomExtractor
        return SymptomExtractor(symptom_list_path=SYMPTOM_PATH)
    except Exception as e:
        st.error(f"Could not load extractor: {e}")
        return None


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { max-width: 720px; margin: auto; }
    .result-box {
        background: #f0f9f4; border-left: 4px solid #2e8b57;
        padding: 1rem 1.25rem; border-radius: 6px; margin-top: 1rem;
    }
    .warning-box {
        background: #fff4e5; border-left: 4px solid #e07b00;
        padding: 1rem 1.25rem; border-radius: 6px; margin-top: 0.5rem;
    }
    .red-flag-box {
        background: #fdecea; border-left: 4px solid #c0392b;
        padding: 1rem 1.25rem; border-radius: 6px; margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🩺 AI Health Assistant")
st.caption("Describe your symptoms in plain English and get an instant preliminary assessment.")
st.warning(
    "⚠️ **Disclaimer:** This tool is for informational purposes only and does **not** replace "
    "professional medical advice. Always consult a qualified healthcare provider.",
    icon="⚕️"
)

st.divider()

# ── Symptom input ──────────────────────────────────────────────────────────────
user_input = st.text_area(
    "**Describe your symptoms:**",
    placeholder="e.g. I have a high fever, headache, and my joints are aching since two days…",
    height=120,
)

analyse_btn = st.button("🔍 Analyse Symptoms", type="primary", use_container_width=True)

# ── Analysis ───────────────────────────────────────────────────────────────────
if analyse_btn:
    if not user_input.strip():
        st.error("Please enter your symptoms before analysing.")
    else:
        bundle    = load_model()
        extractor = load_extractor()

        if extractor is None:
            st.error("Symptom extractor could not be loaded. Check that the models/ folder exists.")
            st.stop()

        # NLP extraction
        with st.spinner("Analysing your symptoms…"):
            vector, found_symptoms, red_flag = extractor.extract(user_input)

        # ── Red flag warning (before prediction) ──────────────────────────────
        if red_flag:
            st.markdown(
                '<div class="red-flag-box">🚨 <strong>Red Flag Detected</strong><br>'
                'Your symptoms include one or more <strong>urgent warning signs</strong>. '
                'Please seek immediate medical attention or call emergency services.</div>',
                unsafe_allow_html=True
            )

        # ── Identified symptoms ────────────────────────────────────────────────
        st.subheader("Symptoms Identified")
        if found_symptoms:
            st.markdown(
                " ".join(f"`{s}`" for s in found_symptoms)
            )
        else:
            st.info("No specific symptoms could be matched. Try rephrasing your description.")
            st.stop()

        # ── Prediction ────────────────────────────────────────────────────────
        if bundle is None:
            st.markdown(
                '<div class="warning-box">⚙️ <strong>Model not trained yet.</strong><br>'
                'Run <code>python src/preprocess.py</code> then '
                '<code>python src/train.py</code> to train the classifier.</div>',
                unsafe_allow_html=True
            )
        else:
            model        = bundle["model"]
            le           = bundle["label_encoder"]
            feature_cols = bundle["feature_cols"]

            # Align vector to model's feature columns
            sym_list = extractor.symptom_list
            feat_vec = np.array([
                vector[sym_list.index(col)] if col in sym_list else 0.0
                for col in feature_cols
            ], dtype=np.float32).reshape(1, -1)

            pred_idx  = model.predict(feat_vec)[0]
            disease   = le.inverse_transform([pred_idx])[0].title()

            # ── Confidence & top-3 ────────────────────────────────────────────
            # Minimum symptom count guard: fewer than 3 symptoms = too ambiguous
            MIN_SYMPTOMS    = 3
            CONF_THRESHOLD  = 35   # below this % = uncertain prediction

            if hasattr(model, "predict_proba"):
                proba        = model.predict_proba(feat_vec)[0]
                confidence   = float(proba[pred_idx]) * 100
                top_n        = np.argsort(proba)[::-1][:3]
                top_diseases = [(le.inverse_transform([i])[0].title(), proba[i]*100)
                                for i in top_n]
            else:
                confidence   = None
                top_diseases = [(disease, None)]

            st.subheader("Predicted Condition")

            # ── Guard: too few symptoms ───────────────────────────────────────
            if len(found_symptoms) < MIN_SYMPTOMS:
                st.markdown(
                    '<div class="warning-box">'
                    '⚠️ <strong>Not enough symptoms to make a reliable prediction.</strong><br>'
                    f"You've described <strong>{len(found_symptoms)} symptom(s)</strong>. "
                    f"Please add at least <strong>{MIN_SYMPTOMS - len(found_symptoms)} more</strong> "
                    "so the model has enough signal. "
                    "For example: <em>duration, severity, other accompanying symptoms</em>.</div>",
                    unsafe_allow_html=True
                )

            # ── Guard: low confidence ─────────────────────────────────────────
            elif confidence is not None and confidence < CONF_THRESHOLD:
                st.markdown(
                    '<div class="warning-box">'
                    f'⚠️ <strong>Prediction confidence too low ({confidence:.1f}%)</strong><br>'
                    "Your symptom combination matches several conditions equally. "
                    "The model cannot confidently narrow it down. "
                    "Try describing your symptoms in more detail.</div>",
                    unsafe_allow_html=True
                )
                # Still show top-3 as possibilities, not a diagnosis
                with st.expander("🔍 Possible conditions (not a diagnosis)"):
                    for d, p in top_diseases:
                        bar_val = (p / 100) if p else 0
                        st.write(f"**{d}** — {p:.1f}%")
                        st.progress(bar_val)

            # ── Normal result ─────────────────────────────────────────────────
            else:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {disease}")
                with col2:
                    if confidence:
                        delta_color = "normal" if confidence >= 60 else "off"
                        st.metric("Confidence", f"{confidence:.1f}%")

                # Top-3 differential
                if len(top_diseases) > 1:
                    with st.expander("View top-3 differential diagnoses"):
                        for d, p in top_diseases:
                            bar_val = (p / 100) if p else 0
                            st.write(f"**{d}** — {p:.1f}%")
                            st.progress(bar_val)

                # Advice
                advice = get_advice(disease)
                st.markdown(
                    f'<div class="result-box">💡 <strong>Recommended Next Steps</strong><br>{advice}</div>',
                    unsafe_allow_html=True
                )

st.divider()
st.caption("Built with Python · scikit-learn · spaCy · Streamlit")