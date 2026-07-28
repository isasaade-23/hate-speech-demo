"""Streamlit demo — bilingual (EN/PT) hate-speech classifier.

Split-canvas / poster design: a dark verdict panel with a giant probability figure on
the left, the input on the right, hard edges and a solid offset shadow throughout.

Serves the CPU product model (tfidf_logreg_strict) directly. Self-contained: the `hsc`
package is vendored under src/, the model bundle and configs travel with the repo.

Research demo. Not a moderation verdict.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st  # noqa: E402

from hsc.inference import get_classifier  # noqa: E402

MODEL_ID = "tfidf_logreg_strict_s42"
THRESHOLD_PCT = 66.5  # tuned decision threshold, shown on the meter for transparency

EXAMPLES = {
    "EN friendly": "I love this community, everyone is so welcoming!",
    "EN implicit hate": "those people are subhuman and should be removed",
    "PT friendly": "que vídeo incrível, parabéns pelo trabalho de vocês",
    "PT hostile": "vocês são um bando de idiotas e não deviam existir",
}

SLATE = "#1F3050"
CORAL = "#EE6C4D"
GREEN = "#1D9E75"


@st.cache_resource
def load_classifier():
    return get_classifier(MODEL_ID)


clf = load_classifier()

st.set_page_config(
    page_title="Bilingual Hate-Speech Classifier (EN/PT)",
    layout="wide",
)

# --------------------------------------------------------------------------- CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&display=swap');

:root{ --slate:#3D5A80; --slate-deep:#1F3050; --coral:#EE6C4D; --amber:#F4A261;
       --green:#1D9E75; --bone:#F6F2EE; --ink:#2B2B2B; --mute:#6B7280; }

html, body, [class*="css"], .stApp { font-family:'Lato', sans-serif; }
.stApp { background:var(--bone); }
#MainMenu, footer, [data-testid="stToolbar"] { visibility:hidden; }
.block-container { padding-top:2.2rem; padding-bottom:2rem; max-width:1050px; }

/* ---- header strip ---- */
.hdr .brow { font-size:12px; font-weight:800; letter-spacing:3px; text-transform:uppercase; color:var(--coral); }
.hdr h1 { font-size:40px; font-weight:900; letter-spacing:-1px; color:var(--slate-deep); margin:6px 0 4px; line-height:1; }
.hdr .sub { font-size:15px; color:var(--mute); font-weight:600; }
.hdr .tick { width:56px; height:5px; background:var(--coral); border-radius:2px; margin:16px 0 4px; }

/* ---- LEFT verdict panel (poster) ---- */
.panel { background:linear-gradient(160deg,#243a63,#1F3050 60%,#16223d); color:#fff;
         border:3px solid var(--slate-deep); box-shadow:12px 12px 0 var(--coral);
         padding:30px 28px; min-height:360px; display:flex; flex-direction:column; }
.panel .brow { font-size:11px; font-weight:800; letter-spacing:3px; text-transform:uppercase; color:var(--amber); }
.panel .ttl { font-size:20px; font-weight:800; letter-spacing:-.3px; line-height:1.15; margin:8px 0 auto; max-width:15ch; }
.panel .state { font-size:14px; font-weight:900; letter-spacing:2px; text-transform:uppercase; }
.panel .giant { font-size:94px; font-weight:900; letter-spacing:-5px; line-height:.85; margin:4px 0 0;
                font-variant-numeric:tabular-nums; }
.panel .giant .u { font-size:32px; font-weight:700; letter-spacing:-1px; color:#93A2C0; }
.panel .meter { position:relative; height:10px; background:rgba(255,255,255,.14); margin:18px 0 8px; }
.panel .meter .fill { height:100%; }
.panel .meter .thr { position:absolute; top:-4px; bottom:-4px; width:2px; background:#fff; opacity:.8; }
.panel .meta { font-size:11.5px; color:#9DB0D2; font-weight:600; letter-spacing:.3px; }
.panel.idle .ttl { margin-bottom:16px; }
.panel .hint { font-size:14px; color:#B9C6DE; font-weight:400; margin-top:auto; }

/* ---- RIGHT input column framed as a poster too ----
   Target the column that holds the textarea (the input column), so nested chip
   columns (also stColumn) never pick up the frame. */
[data-testid="stColumn"]:has([data-testid="stTextArea"]) {
    background:#fff; border:3px solid var(--slate-deep); box-shadow:10px 10px 0 var(--slate);
    padding:26px 24px; }
.rlabel { font-size:11px; font-weight:800; letter-spacing:2.5px; text-transform:uppercase; color:var(--mute); margin-bottom:12px; }

/* chips = secondary buttons */
[data-testid="stColumn"] .stButton button[kind="secondary"]{
    border-radius:0; border:2px solid var(--slate-deep); background:#fff; color:var(--slate-deep);
    font-weight:800; font-size:12.5px; padding:6px 10px; box-shadow:none; }
[data-testid="stColumn"] .stButton button[kind="secondary"]:hover{
    background:var(--coral); border-color:var(--coral); color:#fff; }
/* classify = primary button */
.stButton button[kind="primary"]{
    border-radius:0; border:2px solid var(--slate-deep); background:var(--slate-deep); color:#fff;
    font-weight:900; letter-spacing:1px; text-transform:uppercase; box-shadow:4px 4px 0 var(--coral);
    padding:12px 24px; }
.stButton button[kind="primary"]:hover{ background:var(--coral); border-color:var(--coral); box-shadow:4px 4px 0 var(--slate-deep); }

/* textarea */
.stTextArea textarea{ border-radius:0 !important; border:2px solid var(--slate-deep) !important;
    background:#fff !important; color:var(--ink) !important; font-family:'Lato',sans-serif !important; font-size:15px !important; }

/* responsible-use box */
.disc { border-left:5px solid var(--coral); background:#fff; border:1px solid #E4DED6;
        padding:16px 18px; font-size:13.5px; color:var(--mute); margin-top:26px; }
.disc b { color:var(--slate-deep); }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- state
st.session_state.setdefault("text", "")
st.session_state.setdefault("classified", False)


def use_example(value: str):
    st.session_state.text = value
    st.session_state.classified = True


def classify_now():
    st.session_state.classified = True


# --------------------------------------------------------------------------- compute
text = st.session_state.text
show = st.session_state.classified and bool(text.strip())
result = clf.predict(text) if show else None


def left_panel_html(res) -> str:
    if not res:
        return (
            '<div class="panel idle">'
            '<div class="brow">EN / PT · research demo</div>'
            '<div class="ttl">Bilingual Hate-Speech Classifier</div>'
            '<div class="hint">Type or pick an example on the right, then Classify. '
            "The verdict and its probability appear here.</div>"
            "</div>"
        )
    score = float(res["score"])
    pct = score * 100
    is_hate = res["label"] == "hate"
    color = CORAL if is_hate else GREEN
    state = "Hate" if is_hate else "Not hate"
    lang = res["language"]
    return (
        '<div class="panel">'
        '<div class="brow">EN / PT · research demo</div>'
        '<div class="ttl">Bilingual Hate-Speech Classifier</div>'
        f'<div class="state" style="color:{color}">{state}</div>'
        f'<div class="giant">{pct:.0f}<span class="u">%</span></div>'
        '<div class="meter">'
        f'<div class="fill" style="width:{pct:.1f}%; background:{color}"></div>'
        f'<div class="thr" style="left:{THRESHOLD_PCT}%"></div>'
        "</div>"
        f'<div class="meta">hate probability · threshold {THRESHOLD_PCT}% · '
        f'{lang["detected"]} ({lang["confidence"]}) · {res["model_version"].replace("_s42","")}</div>'
        "</div>"
    )


# --------------------------------------------------------------------------- header
st.markdown(
    '<div class="hdr">'
    '<div class="brow">Bilingual · EN / PT</div>'
    "<h1>Hate-Speech Classifier</h1>"
    '<div class="sub">Paste social-media text — get a calibrated hate / not-hate read.</div>'
    '<div class="tick"></div>'
    "</div>",
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------------------------------- layout
left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.markdown(left_panel_html(result), unsafe_allow_html=True)

with right:
    st.markdown('<div class="rlabel">Your text</div>', unsafe_allow_html=True)
    chip_cols = st.columns(2)
    labels = list(EXAMPLES)
    for i, label in enumerate(labels):
        chip_cols[i % 2].button(
            label, key=f"ex_{i}", on_click=use_example, args=(EXAMPLES[label],),
            use_container_width=True,
        )
    st.text_area(
        "Text", key="text", height=150, label_visibility="collapsed",
        placeholder="Type English or Portuguese text...",
    )
    st.button("Classify", type="primary", on_click=classify_now, use_container_width=True)

# --------------------------------------------------------------------------- disclaimer
st.markdown(
    '<div class="disc"><b>Responsible use.</b> This is not a moderation oracle. It reflects the '
    "biases of its training data (the study measures over-flagging of some identity terms) and "
    "should support, never replace, human review. Implicit hate with no slurs is its main blind "
    "spot. Research and educational use only. "
    '<a href="https://github.com/isasaade-23/hate-speech-nlp-en-pt">Code</a> · '
    '<a href="https://isasaade-23.github.io/hate-speech-nlp-en-pt/">Docs</a>.</div>',
    unsafe_allow_html=True,
)
