"""Streamlit demo — bilingual (EN/PT) hate-speech classifier.

Serves the CPU product model (tfidf_logreg_strict) directly. Self-contained: the
`hsc` package is vendored under src/, the model bundle and configs travel with the
repo, and lingua bundles its own language models — nothing is fetched at runtime.

Research demo. Not a moderation verdict.
"""

from __future__ import annotations

import os
import sys

# Vendored package lives under src/ (keeps hsc.config.project_root() == repo root,
# so configs/ and models/ resolve correctly).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st  # noqa: E402

from hsc.inference import get_classifier  # noqa: E402

MODEL_ID = "tfidf_logreg_strict_s42"


@st.cache_resource
def load_classifier():
    return get_classifier(MODEL_ID)


clf = load_classifier()

EXAMPLES = {
    "— pick an example —": "",
    "EN — friendly": "I love this community, everyone is so welcoming!",
    "EN — implicit hate": "those people are subhuman and should be removed",
    "PT — friendly": "Você é uma pessoa incrível, obrigado por tudo.",
    "PT — hostile": "vocês são um bando de idiotas e não deviam existir",
}

st.set_page_config(page_title="Bilingual Hate-Speech Classifier (EN/PT)", layout="centered")

st.title("Bilingual Hate-Speech Classifier (EN / PT)")
st.markdown(
    "A probabilistic research classifier for English and Portuguese social-media text. "
    "It returns a hate / not-hate label with a confidence score and the detected language.\n\n"
    "This demo serves the lightweight CPU product model (TF-IDF + Logistic Regression) from a "
    "study comparing classical models with transformers under one leakage-safe protocol. "
    "[Code & results](https://github.com/isasaade-23/hate-speech-nlp-en-pt) · "
    "[Documentation](https://isasaade-23.github.io/hate-speech-nlp-en-pt/)."
)

choice = st.selectbox("Try an example", list(EXAMPLES), index=0)
text = st.text_area(
    "Text",
    value=EXAMPLES[choice],
    height=140,
    placeholder="Type English or Portuguese text...",
)

if st.button("Classify", type="primary") and text.strip():
    p = clf.predict(text)
    score = p["score"]
    is_hate = p["label"] == "hate"
    col1, col2 = st.columns(2)
    col1.metric("Prediction", "HATE" if is_hate else "NOT HATE")
    col2.metric("Hate probability", f"{score:.1%}")
    st.progress(min(max(score, 0.0), 1.0))
    lang = p["language"]
    st.caption(
        f"Detected language: **{lang['detected']}** (confidence {lang['confidence']}) · "
        f"model `{p['model_version']}`"
    )

st.divider()
st.caption(
    "Responsible use: this is not a moderation oracle. It reflects the biases of its training "
    "data (the study measures over-flagging of some identity terms) and should support, never "
    "replace, human review. Implicit hate with no slurs is its main blind spot. Research and "
    "educational use only."
)
