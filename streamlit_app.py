"""Streamlit demo — bilingual (EN/PT) hate-speech classifier.

A poster / split-canvas design: an editorial landing (how it was built, results, links)
scrolls into the live tool — a dark verdict panel with a giant probability figure on the
left, the input on the right, hard edges and solid offset shadows throughout.

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

REPO = "https://github.com/isasaade-23/hate-speech-nlp-en-pt"
DOCS = "https://isasaade-23.github.io/hate-speech-nlp-en-pt/"
DEMO_REPO = "https://github.com/isasaade-23/hate-speech-demo"

CORAL = "#EE6C4D"
GREEN = "#1D9E75"


@st.cache_resource
def load_classifier():
    return get_classifier(MODEL_ID)


clf = load_classifier()

st.set_page_config(page_title="Bilingual Hate-Speech Classifier (EN/PT)", layout="wide")

# --------------------------------------------------------------------------- CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&display=swap');

:root{ --slate:#3D5A80; --slate-deep:#1F3050; --coral:#EE6C4D; --amber:#F4A261;
       --green:#1D9E75; --bone:#F6F2EE; --ink:#2B2B2B; --mute:#6B7280; --line:#DAD2C7; }

html, body, [class*="css"], .stApp { font-family:'Lato', sans-serif; }
.stApp { background:var(--bone); }

/* kill the white Streamlit top bar that overlapped the content */
header[data-testid="stHeader"]{ display:none; }
#MainMenu, footer, [data-testid="stToolbar"]{ display:none; }
.block-container{ padding-top:1.4rem; padding-bottom:2rem; max-width:1050px; }

/* ================= animations (CSS only, safe fallbacks) ================= */
@keyframes fadeUp { from{ opacity:0; transform:translateY(18px); } to{ opacity:1; transform:none; } }
@media (prefers-reduced-motion: no-preference){
  .anim{ animation:fadeUp .7s cubic-bezier(.2,.7,.2,1) both; }
  .d1{ animation-delay:.05s } .d2{ animation-delay:.14s } .d3{ animation-delay:.23s } .d4{ animation-delay:.32s }
  @supports (animation-timeline: view()){
    .reveal{ animation:fadeUp both; animation-timeline:view(); animation-range:entry 4% cover 26%; }
  }
}

/* ================= landing ================= */
.land{ margin:6px 0 8px; }
.land .brow{ font-size:12px; font-weight:800; letter-spacing:3px; text-transform:uppercase; color:var(--coral); }
.land h1{ font-size:clamp(34px,6vw,60px); font-weight:900; letter-spacing:-1.5px; line-height:.98;
          color:var(--slate-deep); margin:12px 0 0; max-width:16ch; text-wrap:balance; }
.land .tag{ font-size:17px; color:var(--mute); font-weight:400; max-width:52ch; margin:16px 0 0; }
.land .tick{ width:64px; height:6px; background:var(--coral); border-radius:2px; margin:22px 0 0; }

.hero{ margin:30px 0 0; background:linear-gradient(160deg,#243a63,#1F3050 60%,#16223d); color:#fff;
       border:3px solid var(--slate-deep); box-shadow:12px 12px 0 var(--coral); padding:26px 28px;
       display:flex; align-items:center; gap:26px; flex-wrap:wrap; }
.hero .klabel{ font-size:11px; font-weight:800; letter-spacing:2.5px; text-transform:uppercase; color:var(--amber); }
.hero .kmain{ font-size:26px; font-weight:900; letter-spacing:-.5px; margin:6px 0 0; line-height:1.1; }
.hero .kmain b{ color:var(--coral); }
.hero .ksub{ font-size:14px; color:#B9C6DE; margin:8px 0 0; max-width:46ch; }
.hero .knum{ margin-left:auto; text-align:right; }
.hero .knum .big{ font-size:60px; font-weight:900; letter-spacing:-3px; line-height:.9;
                  font-variant-numeric:tabular-nums;
                  background:linear-gradient(90deg,#F4A261,#EE6C4D); -webkit-background-clip:text;
                  background-clip:text; -webkit-text-fill-color:transparent; }
.hero .knum .cap{ font-size:11px; color:#9DB0D2; font-weight:700; letter-spacing:1px; text-transform:uppercase; }

.seclabel{ font-size:12px; font-weight:800; letter-spacing:3px; text-transform:uppercase; color:var(--slate);
           border-bottom:2px solid var(--line); padding-bottom:8px; margin:46px 0 22px; }

.steps{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
@media (max-width:760px){ .steps{ grid-template-columns:repeat(2,1fr); } }
@media (max-width:460px){ .steps{ grid-template-columns:1fr; } }
.step{ background:#fff; border:2px solid var(--slate-deep); box-shadow:6px 6px 0 var(--slate);
       padding:16px 16px 18px; }
.step .n{ font-family:'Lato'; font-size:13px; font-weight:900; color:#fff; background:var(--slate-deep);
          display:inline-block; padding:2px 9px; letter-spacing:1px; }
.step h4{ font-size:16px; font-weight:900; color:var(--slate-deep); margin:12px 0 6px; letter-spacing:-.2px; }
.step p{ font-size:13px; color:var(--mute); line-height:1.55; margin:0; }
.step.coral{ box-shadow:6px 6px 0 var(--coral); }
.step.coral .n{ background:var(--coral); }
.step.coral h4{ color:var(--coral); }
/* 05 Ship spans the full width of the four steps above, as a horizontal band */
.step.ship{ grid-column:1 / -1; display:flex; align-items:center; gap:24px; padding:18px 20px; }
.step.ship .shiphead{ display:flex; align-items:center; gap:12px; flex:none; }
.step.ship h4{ margin:0; font-size:19px; }
.step.ship p{ font-size:14px; }
@media (max-width:460px){ .step.ship{ flex-direction:column; align-items:flex-start; gap:10px; } }

.nums{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; }
.numc{ background:var(--slate-deep); color:#fff; border:3px solid var(--slate-deep);
       box-shadow:8px 8px 0 var(--coral); padding:20px 22px; }
.numc .v{ font-size:44px; font-weight:900; letter-spacing:-2px; line-height:1; font-variant-numeric:tabular-nums; }
.numc .k{ font-size:12.5px; color:#B9C6DE; font-weight:700; margin-top:8px; line-height:1.4; }
.numc.alt{ background:#fff; color:var(--slate-deep); box-shadow:8px 8px 0 var(--slate); }
.numc.alt .k{ color:var(--mute); }

.links{ display:flex; gap:12px; flex-wrap:wrap; margin:26px 0 0; }
.links a{ text-decoration:none; font-weight:800; font-size:13.5px; letter-spacing:.5px; text-transform:uppercase;
          padding:11px 22px; border:2px solid var(--slate-deep); color:var(--slate-deep); background:#fff; }
.links a:hover{ background:var(--slate-deep); color:#fff; }
.links a.primary{ background:var(--coral); border-color:var(--coral); color:#fff; box-shadow:4px 4px 0 var(--slate-deep); }
.links a.primary:hover{ background:var(--slate-deep); border-color:var(--slate-deep); box-shadow:4px 4px 0 var(--coral); }

.cta{ margin:52px 0 6px; text-align:center; }
.cta .line{ height:2px; background:var(--line); margin-bottom:18px; }
.cta .go{ font-size:13px; font-weight:800; letter-spacing:3px; text-transform:uppercase; color:var(--coral); }
.cta .go .arrow{ display:block; font-size:22px; color:var(--slate); margin-top:4px; }

/* ================= tool: LEFT verdict panel ================= */
.panel{ background:linear-gradient(160deg,#243a63,#1F3050 60%,#16223d); color:#fff;
        border:3px solid var(--slate-deep); box-shadow:12px 12px 0 var(--coral);
        padding:30px 28px; min-height:360px; display:flex; flex-direction:column; }
.panel .brow{ font-size:11px; font-weight:800; letter-spacing:3px; text-transform:uppercase; color:var(--amber); }
.panel .ttl{ font-size:20px; font-weight:800; letter-spacing:-.3px; line-height:1.15; margin:8px 0 auto; max-width:15ch; }
.panel .state{ font-size:14px; font-weight:900; letter-spacing:2px; text-transform:uppercase; }
.panel .giant{ font-size:94px; font-weight:900; letter-spacing:-5px; line-height:.85; margin:4px 0 0;
               font-variant-numeric:tabular-nums; }
.panel .giant .u{ font-size:32px; font-weight:700; letter-spacing:-1px; color:#93A2C0; }
.panel .meter{ position:relative; height:10px; background:rgba(255,255,255,.14); margin:18px 0 8px; }
.panel .meter .fill{ height:100%; }
.panel .meter .thr{ position:absolute; top:-4px; bottom:-4px; width:2px; background:#fff; opacity:.8; }
.panel .meta{ font-size:11.5px; color:#9DB0D2; font-weight:600; letter-spacing:.3px; }
.panel.idle .ttl{ margin-bottom:16px; }
.panel .hint{ font-size:14px; color:#B9C6DE; font-weight:400; margin-top:auto; }

/* tool: RIGHT input column framed as a poster (target the column with the textarea) */
[data-testid="stColumn"]:has([data-testid="stTextArea"]){
    background:#fff; border:3px solid var(--slate-deep); box-shadow:10px 10px 0 var(--slate);
    padding:26px 24px; }
.rlabel{ font-size:11px; font-weight:800; letter-spacing:2.5px; text-transform:uppercase; color:var(--mute); margin-bottom:12px; }

[data-testid="stColumn"] .stButton button[kind="secondary"]{
    border-radius:0; border:2px solid var(--slate-deep); background:#fff; color:var(--slate-deep);
    font-weight:800; font-size:12.5px; padding:6px 10px; box-shadow:none; }
[data-testid="stColumn"] .stButton button[kind="secondary"]:hover{
    background:var(--coral); border-color:var(--coral); color:#fff; }
.stButton button[kind="primary"]{
    border-radius:0; border:2px solid var(--slate-deep); background:var(--slate-deep); color:#fff;
    font-weight:900; letter-spacing:1px; text-transform:uppercase; box-shadow:4px 4px 0 var(--coral); padding:12px 24px; }
.stButton button[kind="primary"]:hover{ background:var(--coral); border-color:var(--coral); box-shadow:4px 4px 0 var(--slate-deep); }

.stTextArea textarea{ border-radius:0 !important; border:2px solid var(--slate-deep) !important;
    background:#fff !important; color:var(--ink) !important; font-family:'Lato',sans-serif !important; font-size:15px !important; }

.disc{ border-left:5px solid var(--coral); background:#fff; border:1px solid var(--line);
       padding:16px 18px; font-size:13.5px; color:var(--mute); margin-top:26px; }
.disc b{ color:var(--slate-deep); }
.disc a{ color:var(--slate); font-weight:700; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- landing
st.markdown(
    f"""
<div class="land">
  <div class="brow anim d1">Research demo · EN / PT</div>
  <h1 class="anim d2">Bilingual Hate-Speech Detection</h1>
  <p class="tag anim d3">A reproducible study comparing classical models with transformers under one
     leakage-safe protocol — and the live, calibrated classifier you can try below.</p>
  <div class="tick anim d4"></div>

  <div class="hero anim d4">
    <div>
      <div class="klabel">The finding</div>
      <div class="kmain">Transformers win — and it's <b>statistically significant</b></div>
      <div class="ksub">XLM-R reaches macro-F1 0.750 vs 0.709 for the best classical baseline
        (paired McNemar + Holm). This demo serves the 3.6 MB classical MVP — within ~4 points, no GPU.</div>
    </div>
    <div class="knum">
      <div class="big">0.750</div>
      <div class="cap">best macro-F1</div>
    </div>
  </div>

  <div class="seclabel reveal">How it was built</div>
  <div class="steps">
    <div class="step reveal"><span class="n">01</span><h4>Harmonize</h4>
      <p>Four Kaggle datasets — EN tweets, EN memes (OCR), PT comments — folded into one binary
         schema under strict and broad label policies.</p></div>
    <div class="step reveal"><span class="n">02</span><h4>De-leak</h4>
      <p>Exact + near-duplicate (MinHash/LSH) dedup, then a group-stratified frozen split so no
         paraphrase crosses train and test. Enforced by a CI test.</p></div>
    <div class="step reveal"><span class="n">03</span><h4>Train</h4>
      <p>TF-IDF and multilingual SBERT → LogReg / SVM / LightGBM (local, CPU); XLM-R, BERTimbau and
         BERTweet fine-tuned on Colab GPU.</p></div>
    <div class="step reveal"><span class="n">04</span><h4>Evaluate</h4>
      <p>Macro-F1, paired McNemar + Holm, probability calibration (ECE), an identity-term bias probe,
         and cross-lingual transfer.</p></div>
    <div class="step ship coral reveal">
      <div class="shiphead"><span class="n">05</span><h4>Ship</h4></div>
      <p>Pareto pick for the product: <b>tfidf_logreg</b> — 3.6 MB, ~1.6 ms on CPU. That's the model
         answering you below.</p></div>
  </div>

  <div class="seclabel reveal">Results</div>
  <div class="nums">
    <div class="numc reveal"><div class="v">0.750</div><div class="k">best transformer<br>(XLM-R, macro-F1)</div></div>
    <div class="numc alt reveal"><div class="v">0.709</div><div class="k">this demo<br>(classical MVP, macro-F1)</div></div>
    <div class="numc reveal"><div class="v">0.42→0.63</div><div class="k">EN→PT zero-shot transfer<br>(TF-IDF → multilingual SBERT)</div></div>
  </div>

  <div class="links reveal">
    <a class="primary" href="{REPO}" target="_blank">Code &amp; study</a>
    <a href="{DOCS}" target="_blank">Documentation</a>
    <a href="{DEMO_REPO}" target="_blank">Demo source</a>
  </div>

  <div class="cta reveal">
    <div class="line"></div>
    <div class="go">Try it live<span class="arrow">&#8595;</span></div>
  </div>
</div>
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


# --------------------------------------------------------------------------- tool
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
    f'<a href="{REPO}">Code</a> · <a href="{DOCS}">Docs</a>.</div>',
    unsafe_allow_html=True,
)
