"""Streamlit demo — bilingual (EN/PT) hate-speech classifier.

Luciola visual system over Streamlit: light + language + AI. Streamlit provides the
widgets and reactivity; a token-based design system (color roles, spacing scale,
surface levels, one icon family, firefly-constellation texture) turns the generated
containers into a designed product: header > hero > method > results > tool > footer.

Serves the CPU product model (tfidf_logreg_strict) directly. Self-contained: the `hsc`
package is vendored under src/, the model bundle and configs travel with the repo.

Research demo. Not a moderation verdict.
"""

from __future__ import annotations

import base64
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from hsc.clean import clean_text  # noqa: E402
from hsc.inference import get_classifier  # noqa: E402

MODEL_ID = "tfidf_logreg_strict_s42"
UNCERTAIN_MARGIN = 0.08  # |score - threshold| below this reads as "near the threshold"

REPO = "https://github.com/isasaade-23/hate-speech-nlp-en-pt"
DOCS = "https://isasaade-23.github.io/hate-speech-nlp-en-pt/"
DEMO_REPO = "https://github.com/isasaade-23/hate-speech-demo"

CORAL = "#EE6C4D"
GREEN = "#1D9E75"
AMBER = "#F4A261"

# brand mark embedded as a data URI (Streamlit sanitizes inline <svg>; a CSS background survives)
_MARK_PATH = Path(__file__).parent / "assets" / "luciola-mark.svg"
MARK_URI = (
    "data:image/svg+xml;base64," + base64.b64encode(_MARK_PATH.read_bytes()).decode()
    if _MARK_PATH.exists()
    else ""
)

# dark-mode token override, injected when the user flips the switch
DARK_CSS = """
<style>
:root{
  --bg:#121a2b; --surface:#1b2740; --surface-2:#16223a; --bd:#38507a; --heading:#EAF0F8;
  --ink:#E4ECF6; --mute:#9DB0D2; --line:#2b3a56; --bone:#121a2b;
  --danger:#FF8A6B; --success:#4FC79A; --warning:#F4B266;
  --shadow-1:0 2px 12px rgba(0,0,0,.35); --shadow-2:0 6px 24px rgba(0,0,0,.45);
}
html, body, .stApp{ background:#121a2b; }
.stApp{
  background-image:
    radial-gradient(420px 300px at 85% 4%, rgba(238,108,77,.05), transparent 70%),
    radial-gradient(360px 260px at 8% 30%, rgba(157,176,210,.05), transparent 70%),
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='560' height='560' viewBox='0 0 560 560'><g fill='none' stroke='%23ABC1E0' stroke-opacity='.05'><path d='M60 90 L200 60 L330 140'/><path d='M330 140 L470 90'/><path d='M120 300 L250 260 L390 330 L510 280'/><path d='M80 470 L230 430 L360 500'/></g><g fill='%23ABC1E0' fill-opacity='.10'><circle cx='60' cy='90' r='2'/><circle cx='200' cy='60' r='1.6'/><circle cx='330' cy='140' r='2.4'/><circle cx='470' cy='90' r='1.5'/><circle cx='120' cy='300' r='1.6'/><circle cx='250' cy='260' r='2'/><circle cx='390' cy='330' r='1.5'/><circle cx='510' cy='280' r='2.2'/><circle cx='80' cy='470' r='1.5'/><circle cx='230' cy='430' r='2.2'/><circle cx='360' cy='500' r='1.6'/></g><g fill='%23EE6C4D' fill-opacity='.10'><circle cx='330' cy='140' r='1.2'/><circle cx='230' cy='430' r='1.2'/></g></svg>");
}
</style>
"""

# example texts are content (kept as-is); only the descriptor labels are translated
EXAMPLES = [
    {"key": "en_ok", "en": "EN friendly", "pt": "EN amigável",
     "text": "I love this community, everyone is so welcoming!"},
    {"key": "en_hate", "en": "EN hate", "pt": "EN ódio",
     "text": "i hate these faggots, they make me sick"},
    {"key": "pt_ok", "en": "PT friendly", "pt": "PT amigável",
     "text": "que vídeo incrível, parabéns pelo trabalho de vocês"},
    {"key": "pt_hate", "en": "PT hate", "pt": "PT ódio",
     "text": "essas mulheres são todas umas vadias nojentas, deviam apanhar"},
]

# --------------------------------------------------------------------------- i18n
T = {
    "en": {
        "eyebrow": "Research demo · EN / PT",
        "title": "Bilingual Hate-Speech Detection",
        "tag": "A study compares classical models with transformers under one leakage-safe "
               "protocol. Below runs the classifier it produced. Try it.",
        "skip": "Skip to the classifier",
        "find_label": "The finding",
        "find_main": "Transformers win. The gap is <b>statistically significant</b>.",
        "find_sub": "BERTimbau reaches macro-F1 0.784. The best classical baseline reaches 0.729 "
                    "(paired McNemar, Holm, p<0.001). This demo runs the classical model, 3.8 MB. It "
                    "stays about five points behind and needs no GPU.",
        "find_num": "0.784", "find_cap": "best macro-F1",
        "how": "How it was built",
        "steps": [
            ("Harmonize", "Five sources become one binary schema. Tweets and memes in English, web "
                          "and Instagram comments in Portuguese. Labels follow two policies, strict and broad."),
            ("De-leak", "Exact and near-duplicate removal (MinHash/LSH) runs first. The group split "
                        "is frozen. No paraphrase crosses train and test."),
            ("Train", "Two regimes. The classical models learn from scratch on our labels, over "
                      "TF-IDF or SBERT features, on CPU. The transformers arrive pretrained on "
                      "billions of words and we fine-tune them on our labels, on Colab GPU."),
            ("Evaluate", "Macro-F1, paired McNemar with Holm, calibration (ECE), an identity-term "
                         "bias probe, cross-lingual transfer."),
            ("Ship", "Pareto picks tfidf_logreg. 3.8 MB, under 5 ms on CPU. It answers you below."),
        ],
        "reads_label": "How it reads text",
        "read1_title": "Word by word", "read1_sub": "bag-of-words",
        "read1_body": "It counts words and character patterns (bag-of-words, TF-IDF), with no "
                      "context beyond short windows. The classifier on top learns from scratch, only "
                      "from our labeled examples. In return it is fast, 3.8 MB, and runs on any CPU.",
        "read1_models": ["Logistic Regression", "Linear SVM", "LightGBM"],
        "read2_title": "The whole sentence", "read2_sub": "contextual",
        "read2_body": "It reads the full sentence in context. These models were pretrained on "
                      "billions of words. We use the multilingual embeddings as they are and "
                      "fine-tune the transformers on our labels. That pretraining is where meaning "
                      "and cross-language transfer come from. It costs more and needs a GPU.",
        "read2_models": ["multilingual SBERT", "XLM-RoBERTa", "BERTimbau", "BERTweet"],
        "tech_label": "Built with",
        "techs": ["NLP", "multilingual EN/PT", "TF-IDF", "transformers", "McNemar + Holm",
                  "calibration", "bias probe"],
        "results": "Results",
        "num1_v": "0.784", "num1_k": "best transformer, strict<br>(BERTimbau, macro-F1)",
        "num2_v": "0.729", "num2_k": "this demo, strict<br>(classical MVP, macro-F1)",
        "num3_v": "0.835", "num3_k": "best model, broad<br>(BERTimbau, macro-F1)",
        "lb_model": "Model", "lb_demo": "TF-IDF + LogReg (this demo)",
        "lb_cap": "Best result per model, test macro-F1. BERTimbau leads both policies; the "
                  "classical model this demo runs stays within reach on CPU.",
        "diag_label": "Why the simple model holds up",
        "diag_body": "Hate detection here is largely lexical. The model's strongest cues are explicit "
                     "slurs and identity attacks in both languages, plus character patterns that catch "
                     "misspellings. That is why this small linear model stays within four points of the "
                     "transformer. It is also why removing stop words barely moves the score, which points "
                     "to the representation as the limit, not the classifier.",
        "totop": "Top",
        "link_code": "Code &amp; study", "link_docs": "Documentation", "link_demo": "Demo source",
        "cta": "Try it live",
        "pipe": [("Text", "i-msg"), ("Language", "i-globe"), ("Model", "i-cpu"),
                 ("Verdict", "i-target"), ("Confidence", "i-gauge"), ("Explanation", "i-search")],
        "panel_brow": "EN / PT · research demo",
        "panel_title": "Bilingual Hate-Speech Classifier",
        "panel_hint": "Type or pick an example, then press Classify. The verdict and its "
                      "probability show here.",
        "chat_you": "your text", "chat_bot": "Luciola",
        "state_hate": "Hate", "state_nothate": "Not hate",
        "state_uncertain": "near the threshold",
        "status_analyzing": "Analyzing...",
        "status_done": "Analysis complete",
        "status_err": "Enter some text first, then press Classify.",
        "meta_prob": "hate probability", "meta_thr": "threshold",
        "dist_label": "class distribution",
        "your_text": "Your text",
        "ex_label": "Examples",
        "placeholder": "Type English or Portuguese text...",
        "classify": "Classify",
        "exp_label": "Under the hood",
        "exp_model": "Model", "exp_model_v": "TF-IDF + Logistic Regression · strict · 3.8 MB",
        "exp_latency": "Latency", "exp_lang": "Detected language",
        "exp_thr": "Decision threshold", "exp_conf": "Confidence",
        "conf_high": "high", "conf_med": "medium", "conf_low": "low, near the threshold",
        "exp_terms": "Strongest cues in this text",
        "exp_toward": "pushes toward hate", "exp_away": "pushes away from hate",
        "exp_none": "No single feature stands out for this text.",
        "exp_note": "The model is linear, so each prediction decomposes exactly into word and "
                    "character features. Fragments with a dot are character n-grams; the dot marks "
                    "a space. This shows what the model reads. It is not a human explanation of "
                    "why a text is hateful.",
        "disc": "<b>Responsible use.</b> This is not a moderation oracle. It carries the biases of "
                "its training data; the study measures over-flagging of some identity terms. It "
                "should support human review, not replace it. Implicit hate without slurs is where "
                "it fails most. Research and educational use only.",
        "disc_code": "Code", "disc_docs": "Docs",
        "theme_dark": "Dark", "theme_light": "Light",
        "abl_label": "Stop words",
        "abl_sub": "We removed prepositions, pronouns and articles from the word features and "
                   "retrained every classical model. Negations stayed. Hover a cell for the exact numbers.",
        "abl_foot": "Color is scaled within each metric so small gaps still show. Character n-grams "
                    "and IDF already down-weight function words. Removing stop words moves almost "
                    "nothing. Frozen group split, seed 42. Stop list in src/hsc/features/stopwords.py.",
    },
    "pt": {
        "eyebrow": "Demo de pesquisa · EN / PT",
        "title": "Detecção Bilíngue de Discurso de Ódio",
        "tag": "Um estudo compara modelos clássicos e transformers sob um protocolo à prova de "
               "vazamento. Abaixo roda o classificador que ele produziu. Teste.",
        "skip": "Pular para o classificador",
        "find_label": "O achado",
        "find_main": "Os transformers vencem. A diferença é <b>estatisticamente significativa</b>.",
        "find_sub": "O BERTimbau chega a macro-F1 0,784. O melhor clássico chega a 0,729 (McNemar "
                    "pareado, Holm, p<0,001). Este demo roda o clássico, 3,8 MB. Fica cerca de cinco "
                    "pontos atrás e dispensa GPU.",
        "find_num": "0,784", "find_cap": "melhor macro-F1",
        "how": "Como foi construído",
        "steps": [
            ("Harmonizar", "Cinco fontes viram um esquema binário. Tweets e memes em inglês, "
                           "comentários web e do Instagram em português. Os rótulos seguem duas políticas, strict e broad."),
            ("Anti-vazamento", "Primeiro remove duplicatas exatas e quase-duplicatas (MinHash/LSH). "
                               "O split por grupo é congelado. Nenhuma paráfrase cruza treino e teste."),
            ("Treinar", "Dois regimes. Os modelos clássicos aprendem do zero com os nossos rótulos, "
                        "sobre features TF-IDF ou SBERT, na CPU. Os transformers chegam pré-treinados "
                        "em bilhões de palavras e nós os ajustamos com os nossos rótulos, na GPU do Colab."),
            ("Avaliar", "Macro-F1, McNemar pareado com Holm, calibração (ECE), sonda de viés por "
                        "termo de identidade, transferência entre idiomas."),
            ("Publicar", "Pareto escolhe o tfidf_logreg. 3,8 MB, abaixo de 5 ms na CPU. É ele que responde "
                         "abaixo."),
        ],
        "reads_label": "Como ela lê o texto",
        "read1_title": "Palavra por palavra", "read1_sub": "saco de palavras",
        "read1_body": "Conta palavras e padrões de caractere (saco de palavras, TF-IDF), sem contexto "
                      "além de janelas curtas. O classificador em cima aprende do zero, só com os nossos "
                      "exemplos rotulados. Em troca, é rápido, ocupa 3,8 MB e roda em qualquer CPU.",
        "read1_models": ["Logistic Regression", "Linear SVM", "LightGBM"],
        "read2_title": "A frase inteira", "read2_sub": "contextual",
        "read2_body": "Lê a frase inteira em contexto. Esses modelos foram pré-treinados em bilhões de "
                      "palavras. Usamos os embeddings multilíngues como estão e ajustamos os transformers "
                      "com os nossos rótulos. É desse pré-treino que vêm o sentido e a transferência entre "
                      "idiomas. Custa mais e precisa de GPU.",
        "read2_models": ["multilingual SBERT", "XLM-RoBERTa", "BERTimbau", "BERTweet"],
        "tech_label": "Feito com",
        "techs": ["NLP", "multilíngue EN/PT", "TF-IDF", "transformers", "McNemar + Holm",
                  "calibração", "sonda de viés"],
        "results": "Resultados",
        "num1_v": "0,784", "num1_k": "melhor transformer, strict<br>(BERTimbau, macro-F1)",
        "num2_v": "0,729", "num2_k": "este demo, strict<br>(MVP clássico, macro-F1)",
        "num3_v": "0,835", "num3_k": "melhor modelo, broad<br>(BERTimbau, macro-F1)",
        "lb_model": "Modelo", "lb_demo": "TF-IDF + LogReg (este demo)",
        "lb_cap": "Melhor resultado por modelo, macro-F1 no teste. O BERTimbau lidera nas duas "
                  "políticas; o clássico que este demo roda fica ao alcance, na CPU.",
        "diag_label": "Por que o modelo simples se segura",
        "diag_body": "Detectar ódio aqui é, em boa parte, léxico. As pistas mais fortes do modelo são "
                     "palavrão e ataque de identidade nos dois idiomas, além de padrões de caractere que "
                     "pegam erros de escrita. Por isso este modelo linear pequeno fica a quatro pontos do "
                     "transformer. E por isso remover palavras vazias quase não muda a nota, o que aponta "
                     "para a representação como o limite, não o classificador.",
        "totop": "Topo",
        "link_code": "Código &amp; estudo", "link_docs": "Documentação", "link_demo": "Código do demo",
        "cta": "Experimente ao vivo",
        "pipe": [("Texto", "i-msg"), ("Idioma", "i-globe"), ("Modelo", "i-cpu"),
                 ("Veredito", "i-target"), ("Confiança", "i-gauge"), ("Explicação", "i-search")],
        "panel_brow": "EN / PT · demo de pesquisa",
        "panel_title": "Classificador Bilíngue de Discurso de Ódio",
        "panel_hint": "Digite ou escolha um exemplo e clique em Classificar. O veredito e a "
                      "probabilidade aparecem aqui.",
        "chat_you": "seu texto", "chat_bot": "Luciola",
        "state_hate": "Ódio", "state_nothate": "Não é ódio",
        "state_uncertain": "perto do limiar",
        "status_analyzing": "Analisando...",
        "status_done": "Análise concluída",
        "status_err": "Digite um texto primeiro e clique em Classificar.",
        "meta_prob": "probabilidade de ódio", "meta_thr": "limiar",
        "dist_label": "distribuição entre classes",
        "your_text": "Seu texto",
        "ex_label": "Exemplos",
        "placeholder": "Digite um texto em inglês ou português...",
        "classify": "Classificar",
        "exp_label": "Por dentro da predição",
        "exp_model": "Modelo", "exp_model_v": "TF-IDF + Regressão Logística · strict · 3,8 MB",
        "exp_latency": "Latência", "exp_lang": "Idioma detectado",
        "exp_thr": "Limiar de decisão", "exp_conf": "Confiança",
        "conf_high": "alta", "conf_med": "média", "conf_low": "baixa, perto do limiar",
        "exp_terms": "Pistas mais fortes neste texto",
        "exp_toward": "empurra para ódio", "exp_away": "empurra para não-ódio",
        "exp_none": "Nenhuma feature isolada se destaca neste texto.",
        "exp_note": "O modelo é linear, então cada predição se decompõe exatamente em features de "
                    "palavra e de caractere. Fragmentos com ponto são n-gramas de caractere; o ponto "
                    "marca um espaço. Isto mostra o que o modelo lê. Não é uma explicação humana de "
                    "por que um texto é odioso.",
        "disc": "<b>Uso responsável.</b> Isto não é um oráculo de moderação. Carrega os vieses dos "
                "dados de treino; o estudo mede a super-marcação de alguns termos de identidade. "
                "Serve para apoiar a revisão humana, não para substituí-la. Falha mais no ódio "
                "implícito, sem palavrão. Uso apenas para pesquisa e educação.",
        "disc_code": "Código", "disc_docs": "Docs",
        "theme_dark": "Escuro", "theme_light": "Claro",
        "abl_label": "Palavras vazias",
        "abl_sub": "Removemos preposições, pronomes e artigos das features de palavra e retreinamos "
                   "cada modelo clássico. As negações ficaram. Passe o mouse numa célula para ver os números.",
        "abl_foot": "A cor é escalada dentro de cada métrica para que diferenças pequenas apareçam. "
                    "Os n-gramas de caractere e o IDF já reduzem o peso das palavras funcionais. "
                    "Remover palavras vazias quase não muda nada. Split por grupo congelado, seed 42. "
                    "Lista em src/hsc/features/stopwords.py.",
    },
}


# --------------------------------------------------------------- ablation heatmap
HEATMAP_TMPL = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700;900&display=swap');
*{box-sizing:border-box}
html,body{margin:0;background:transparent;font-family:'Lato',system-ui,sans-serif;color:__INK__}
.wrap{padding:2px 16px 14px 4px}
.seg{display:inline-flex;border:1px solid __BORDER__;border-radius:999px;overflow:hidden;margin:0 0 16px}
.seg button{font-family:inherit;font-size:12.5px;font-weight:800;letter-spacing:.6px;text-transform:uppercase;padding:8px 15px;border:none;border-right:1px solid __BORDER__;background:__CARD__;color:__INK__;cursor:pointer;transition:background .2s,color .2s}
.seg button:last-child{border-right:none}
.seg button[aria-pressed="true"]{background:__SEGON_BG__;color:__SEGON_TX__}
.seg button:focus-visible{outline:3px solid #EE6C4D;outline-offset:2px}
.card{background:__CARD__;border:1px solid __BORDER__;border-radius:14px;box-shadow:0 2px 12px rgba(31,48,80,.08);padding:14px 16px 12px;max-width:100%;overflow-x:auto}
.grid{display:grid;grid-template-columns:minmax(150px,1.5fr) repeat(3,1fr);gap:6px;min-width:470px}
.hcell{font-size:11px;font-weight:800;letter-spacing:1px;text-transform:uppercase;color:__MUTE__;padding:2px 6px 6px;align-self:end;text-align:center}
.hcell.corner{text-align:left}
.rlab{font-size:13.5px;font-weight:900;color:__INK__;padding:6px 8px;display:flex;flex-direction:column;justify-content:center;transition:color .2s}
.rlab .pol{font-size:10.5px;font-weight:800;letter-spacing:.5px;text-transform:uppercase;color:__POL__;margin-top:2px}
.cell{height:46px;display:flex;align-items:center;justify-content:center;font-family:'SFMono-Regular',Consolas,monospace;font-size:13.5px;font-weight:700;font-variant-numeric:tabular-nums;cursor:pointer;border:2px solid transparent;border-radius:8px;transition:background-color .4s ease,color .4s ease;animation:cellin .45s both}
.cell:hover{border-color:__BORDER__}
@keyframes cellin{from{opacity:0;transform:scale(.86)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){.cell{animation:none}}
.legend{display:flex;align-items:center;gap:10px;margin:12px 2px 0;font-size:11.5px;font-weight:700;color:__MUTE__}
.legbar{height:12px;width:130px;border:1px solid rgba(0,0,0,.15);border-radius:6px}
.take{margin:13px 2px 0;font-size:13px;color:__INK__;line-height:1.5;max-width:66ch}
.take b{color:#EE6C4D}
.tip{position:fixed;pointer-events:none;opacity:0;transform:translate(-50%,-100%);background:#1F3050;color:#fff;border:1px solid rgba(255,255,255,.4);border-radius:10px;box-shadow:0 6px 24px rgba(0,0,0,.35);padding:9px 12px;font-size:12.5px;z-index:30;min-width:180px;transition:opacity .08s}
.tip .tt{font-weight:900;font-size:12.5px;margin-bottom:5px}
.tip .tr{display:flex;justify-content:space-between;gap:16px;font-variant-numeric:tabular-nums;margin-top:3px}
.tip .k{color:#9DB0D2}
.tip .pos{color:#7FE0B0}.tip .neg{color:#FFA98C}.tip .z{color:#B9C6DE}
</style></head><body>
<div class="wrap">
  <div class="seg" role="group" aria-label="view">
    <button data-mode="base" aria-pressed="true">__T_BASE__</button>
    <button data-mode="nostop" aria-pressed="false">__T_NOSTOP__</button>
    <button data-mode="delta" aria-pressed="false">__T_DELTA__</button>
  </div>
  <div class="card"><div class="grid" id="grid"></div></div>
  <div class="legend" id="legend"></div>
  <div class="take" id="take"></div>
</div>
<div class="tip" id="tip" aria-hidden="true"></div>
<script>
const DATA=[
 {model:"LogReg",policy:"strict",f1:[0.7293,0.7247],auc:[0.8531,0.8514],rec:[0.5044,0.5323]},
 {model:"LogReg",policy:"broad",f1:[0.7459,0.7468],auc:[0.8199,0.8206],rec:[0.6558,0.6442]},
 {model:"LightGBM",policy:"strict",f1:[0.7114,0.7057],auc:[0.8380,0.8386],rec:[0.4956,0.4765]},
 {model:"LightGBM",policy:"broad",f1:[0.7246,0.7300],auc:[0.8087,0.8081],rec:[0.5755,0.6346]},
 {model:"SVM",policy:"strict",f1:[0.6967,0.6928],auc:[0.8271,0.8253],rec:[0.4736,0.4428]},
 {model:"SVM",policy:"broad",f1:[0.7288,0.7216],auc:[0.7978,0.7966],rec:[0.6428,0.6625]}
];
const METRICS=[{k:"f1",label:"macro-F1"},{k:"auc",label:"ROC-AUC"},{k:"rec",label:"__REC_LABEL__"}];
const LB={base:"__TT_BASE__",nostop:"__TT_NOSTOP__",delta:"__TT_DELTA__"};
const DELTA_FULL=0.04;
const RANGE={};
METRICS.forEach(function(m){var v=[];DATA.forEach(function(d){v.push(d[m.k][0],d[m.k][1])});RANGE[m.k]=[Math.min.apply(null,v),Math.max.apply(null,v)];});
var mode="base";
function clamp(x,a,b){return Math.max(a,Math.min(b,x))}
function lp(a,b,t){return [Math.round(a[0]+(b[0]-a[0])*t),Math.round(a[1]+(b[1]-a[1])*t),Math.round(a[2]+(b[2]-a[2])*t)]}
function seq(v,k){var r=RANGE[k];var t=r[1]>r[0]?(v-r[0])/(r[1]-r[0]):.5;return lp([__SEQLO__],[__SEQHI__],clamp(t,0,1))}
function dv(d){var t=clamp(d/DELTA_FULL,-1,1);return t>=0?lp([__DIVMID__],[29,158,117],t):lp([__DIVMID__],[238,108,77],-t)}
function css(c){return "rgb("+c[0]+","+c[1]+","+c[2]+")"}
function lum(c){return (0.2126*c[0]+0.7152*c[1]+0.0722*c[2])/255}
function fmt(v){return v.toFixed(3)}
function fmtD(d){return (d>=0?"+":"−")+Math.abs(d).toFixed(3)}
var grid=document.getElementById("grid"),tip=document.getElementById("tip");
function mkd(cls,txt){var d=document.createElement("div");d.className=cls;if(txt!=null)d.textContent=txt;return d}
grid.appendChild(mkd("hcell corner",""));
METRICS.forEach(function(m){grid.appendChild(mkd("hcell",m.label))});
var refs=[];
DATA.forEach(function(d,ri){
  var lab=mkd("rlab");lab.innerHTML='<span>'+d.model+'</span><span class="pol">'+d.policy+'</span>';lab.dataset.row=ri;grid.appendChild(lab);
  var rc=[];
  METRICS.forEach(function(m,ci){
    var c=mkd("cell");c.style.animationDelay=((ri*3+ci)*0.035)+"s";c.dataset.row=ri;
    c.addEventListener("mousemove",function(ev){showTip(ev,ri,m.k)});
    c.addEventListener("mouseenter",function(){setHi(ri,true)});
    c.addEventListener("mouseleave",function(){setHi(ri,false);hideTip()});
    grid.appendChild(c);rc.push(c);
  });
  refs.push(rc);
});
function setHi(ri,on){var ls=document.querySelectorAll(".rlab");for(var i=0;i<ls.length;i++){if(ls[i].dataset.row==ri)ls[i].style.color=on?"#EE6C4D":""}}
function paint(){
  DATA.forEach(function(d,ri){METRICS.forEach(function(m,ci){
    var c=refs[ri][ci],vs=d[m.k],col,txt,val;
    if(mode==="delta"){val=vs[1]-vs[0];col=dv(val);txt=fmtD(val)}
    else{val=mode==="base"?vs[0]:vs[1];col=seq(val,m.k);txt=fmt(val)}
    c.style.background=css(col);c.style.color=lum(col)<0.55?"#fff":"#1F3050";c.textContent=txt;
  })});
  var L=document.getElementById("legend");
  if(mode==="delta"){L.innerHTML='<span>__LEG_WORSE__</span><span class="legbar" style="background:linear-gradient(90deg,#EE6C4D,__MIDHEX__,#1D9E75)"></span><span>__LEG_BETTER__</span>'}
  else{L.innerHTML='<span>__LEG_LO__</span><span class="legbar" style="background:linear-gradient(90deg,__SEQLOHEX__,__SEQHIHEX__)"></span><span>__LEG_HI__</span>'}
  document.getElementById("take").innerHTML=mode==="delta"?"__TAKE_DELTA__":"__TAKE_VAL__";
}
function showTip(ev,ri,mk2){
  var d=DATA[ri],vs=d[mk2],delta=vs[1]-vs[0];
  var dc=Math.abs(delta)<0.001?"z":(delta>0?"pos":"neg");
  var ml=METRICS.filter(function(m){return m.k===mk2})[0].label;
  tip.innerHTML='<div class="tt">'+d.model+' · '+d.policy+' · '+ml+'</div>'+
    '<div class="tr"><span class="k">'+LB.base+'</span><span>'+fmt(vs[0])+'</span></div>'+
    '<div class="tr"><span class="k">'+LB.nostop+'</span><span>'+fmt(vs[1])+'</span></div>'+
    '<div class="tr"><span class="k">'+LB.delta+'</span><span class="'+dc+'">'+fmtD(delta)+'</span></div>';
  tip.style.opacity=1;tip.setAttribute("aria-hidden","false");
  tip.style.left=ev.clientX+"px";tip.style.top=(ev.clientY-14)+"px";
}
function hideTip(){tip.style.opacity=0;tip.setAttribute("aria-hidden","true")}
var btns=document.querySelectorAll(".seg button");
for(var bi=0;bi<btns.length;bi++){btns[bi].addEventListener("click",function(){
  mode=this.dataset.mode;
  for(var j=0;j<btns.length;j++){btns[j].setAttribute("aria-pressed",btns[j]===this?"true":"false")}
  paint();
})}
paint();
</script></body></html>"""

HM = {
    "en": {
        "T_BASE": "Baseline", "T_NOSTOP": "No stop words", "T_DELTA": "Difference",
        "TT_BASE": "Baseline", "TT_NOSTOP": "No stop words", "TT_DELTA": "Δ vs baseline",
        "REC_LABEL": "Recall (hate)",
        "LEG_LO": "lower", "LEG_HI": "higher", "LEG_WORSE": "worse", "LEG_BETTER": "better",
        "TAKE_VAL": "Deeper is a higher score. Switch between Baseline and No stop words. "
                    "<b>The colors barely move.</b>",
        "TAKE_DELTA": "Green is a gain, coral a loss, pale is no change. "
                      "<b>Almost every cell is pale.</b> Removing stop words does not move the models.",
    },
    "pt": {
        "T_BASE": "Base", "T_NOSTOP": "Sem stopwords", "T_DELTA": "Diferença",
        "TT_BASE": "Com stopwords", "TT_NOSTOP": "Sem stopwords", "TT_DELTA": "Δ vs base",
        "REC_LABEL": "Recall (ódio)",
        "LEG_LO": "menor", "LEG_HI": "maior", "LEG_WORSE": "pior", "LEG_BETTER": "melhor",
        "TAKE_VAL": "Mais escuro é nota mais alta. Alterne entre Base e Sem stopwords. "
                    "<b>As cores quase não mudam.</b>",
        "TAKE_DELTA": "Verde é ganho, coral é perda, claro é sem mudança. "
                      "<b>Quase toda célula está clara.</b> Remover palavras vazias não move os modelos.",
    },
}


HM_THEME = {
    False: {
        "INK": "#1F3050", "MUTE": "#6B7280", "POL": "#3D5A80",
        "CARD": "#ffffff", "BORDER": "#DAD2C7", "SHADOW": "#3D5A80",
        "SEGON_BG": "#1F3050", "SEGON_TX": "#ffffff",
        "SEQLO": "233,239,246", "SEQHI": "31,48,80", "DIVMID": "246,242,238",
        "SEQLOHEX": "#E9EFF6", "SEQHIHEX": "#1F3050", "MIDHEX": "#F6F2EE",
    },
    True: {
        "INK": "#EAF0F8", "MUTE": "#9DB0D2", "POL": "#9DB0D2",
        "CARD": "#1b2740", "BORDER": "#2b3a56", "SHADOW": "#24344f",
        "SEGON_BG": "#EAF0F8", "SEGON_TX": "#121a2b",
        "SEQLO": "38,52,80", "SEQHI": "171,193,224", "DIVMID": "27,39,64",
        "SEQLOHEX": "#263450", "SEQHIHEX": "#ABC1E0", "MIDHEX": "#1b2740",
    },
}


def heatmap_html(language: str, dark: bool = False) -> str:
    html = HEATMAP_TMPL
    for key, val in {**HM[language], **HM_THEME[bool(dark)]}.items():
        html = html.replace("__" + key + "__", val)
    return html


def html_block(markup: str) -> str:
    """Strip per-line indentation so st.markdown never reads nested HTML as a
    4-space markdown code block. Content lives inside tags, so this is safe."""
    return re.sub(r"(?m)^[ \t]+", "", markup)


def ico(name: str) -> str:
    """One icon family (outline, 2px stroke) rendered as CSS-mask spans so the color
    always follows currentColor in both themes. Streamlit sanitizes inline <svg>."""
    return f'<span class="ico {name}" aria-hidden="true"></span>'


def render_iframe(html: str, height: int) -> None:
    """Embed self-contained HTML+JS. Prefer st.iframe (current API); fall back to
    the deprecated components.html on older Streamlit."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=False)


@st.cache_resource
def load_classifier():
    return get_classifier(MODEL_ID)


clf = load_classifier()
THRESHOLD_PCT = round(clf.threshold * 100, 1)  # single source of truth: the bundle


@st.cache_resource
def feature_names():
    return clf.vectorizer.get_feature_names_out()


def top_terms(raw_text: str, k: int = 6) -> list[tuple[str, float]]:
    """Exact linear attribution: contribution of feature i = tfidf_i * coef_i.

    Only defined for linear estimators with coef_ (the served LogReg qualifies).
    Returns the k features with the largest absolute contribution, cleaned of
    the FeatureUnion prefix. Spaces inside char n-grams are shown as a dot.
    """
    coef = getattr(clf.estimator, "coef_", None)
    if coef is None:
        return []
    coef = coef.ravel()
    cleaned = clean_text(raw_text, clf._profile)
    X = clf.vectorizer.transform([cleaned])
    names = feature_names()
    contribs = [(int(i), float(X[0, i] * coef[i])) for i in X.nonzero()[1]]
    contribs.sort(key=lambda p: -abs(p[1]))
    out, seen = [], set()
    for i, c in contribs:
        name = str(names[i]).split("__", 1)[-1]
        disp = name.replace(" ", "·")
        if len(disp.strip("·")) < 2 or disp in seen or abs(c) < 1e-6:
            continue
        seen.add(disp)
        out.append((disp, c))
        if len(out) >= k:
            break
    return out


st.set_page_config(page_title="Luciola · Bilingual Hate-Speech Classifier (EN/PT)",
                   page_icon="🔆", layout="wide")

# ------------------------------------------------------------------- design system
# Tokens first (color, spacing, shape, elevation, type), then the icon family, then
# components. Surfaces build depth in five levels: bg -> textured bg -> surface ->
# elevated -> accent (deep navy). Coral and amber are budgeted: actions, verdicts and
# one highlight per section.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&display=swap');

:root{
  /* color tokens */
  --slate:#3D5A80; --slate-deep:#1F3050; --coral:#EE6C4D; --amber:#F4A261;
  --bg:#F6F2EE; --surface:#ffffff; --surface-2:#EFE9E0;
  --ink:#2B2B2B; --heading:#1F3050; --mute:#5D6673;
  --bd:#1F3050; --line:#DAD2C7;
  --primary:#EE6C4D; --danger:#B93A20; --success:#0F7A56; --warning:#8A5A10;
  --bone:#F6F2EE;
  /* spacing scale */
  --s1:4px; --s2:8px; --s3:16px; --s4:24px; --s5:32px; --s6:48px; --s7:64px;
  /* shape + elevation */
  --r-card:14px; --r-pill:999px; --r-input:12px;
  --shadow-1:0 2px 12px rgba(31,48,80,.07);
  --shadow-2:0 8px 28px rgba(31,48,80,.13);
}

html, body, [class*="css"], .stApp { font-family:'Lato', sans-serif; }
.stApp { background:var(--bg); }
/* level 2: firefly-constellation texture, quiet enough to only appear when sought */
.stApp{
  background-image:
    radial-gradient(420px 300px at 85% 4%, rgba(238,108,77,.045), transparent 70%),
    radial-gradient(360px 260px at 8% 30%, rgba(61,90,128,.05), transparent 70%),
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='560' height='560' viewBox='0 0 560 560'><g fill='none' stroke='%233D5A80' stroke-opacity='.06'><path d='M60 90 L200 60 L330 140'/><path d='M330 140 L470 90'/><path d='M120 300 L250 260 L390 330 L510 280'/><path d='M80 470 L230 430 L360 500'/></g><g fill='%233D5A80' fill-opacity='.12'><circle cx='60' cy='90' r='2'/><circle cx='200' cy='60' r='1.6'/><circle cx='330' cy='140' r='2.4'/><circle cx='470' cy='90' r='1.5'/><circle cx='120' cy='300' r='1.6'/><circle cx='250' cy='260' r='2'/><circle cx='390' cy='330' r='1.5'/><circle cx='510' cy='280' r='2.2'/><circle cx='80' cy='470' r='1.5'/><circle cx='230' cy='430' r='2.2'/><circle cx='360' cy='500' r='1.6'/></g><g fill='%23EE6C4D' fill-opacity='.12'><circle cx='330' cy='140' r='1.2'/><circle cx='230' cy='430' r='1.2'/></g></svg>");
}
header[data-testid="stHeader"]{ display:none; }
#MainMenu, footer, [data-testid="stToolbar"]{ display:none; }
.block-container{ padding-top:var(--s3); padding-bottom:var(--s5); max-width:1050px; }
[data-testid="stVerticalBlock"]{ gap:.7rem; }

/* icon family: outline, 2px stroke, colored via currentColor (CSS mask) */
.ico{ display:inline-block; width:15px; height:15px; background:currentColor; flex:none;
  vertical-align:-2px;
  -webkit-mask:var(--i) center/contain no-repeat; mask:var(--i) center/contain no-repeat; }
.i-fly{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'><circle cx='12' cy='12' r='3'/><path d='M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.5 1.5M16.9 16.9l1.5 1.5M18.4 5.6l-1.5 1.5M7.1 16.9l-1.5 1.5'/></svg>");}
.i-globe{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'><circle cx='12' cy='12' r='10'/><path d='M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z'/></svg>");}
.i-cpu{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'><rect x='4' y='4' width='16' height='16' rx='2'/><rect x='9' y='9' width='6' height='6'/><path d='M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3'/></svg>");}
.i-gauge{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'><path d='M12 14l4-4'/><path d='M3.34 19a10 10 0 1 1 17.32 0'/></svg>");}
.i-zap{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linejoin='round'><path d='M13 2L3 14h9l-1 8 10-12h-9l1-8z'/></svg>");}
.i-target{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2'><circle cx='12' cy='12' r='10'/><circle cx='12' cy='12' r='6'/><circle cx='12' cy='12' r='2'/></svg>");}
.i-search{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'><circle cx='11' cy='11' r='8'/><path d='M21 21l-4.35-4.35'/></svg>");}
.i-alert{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z'/><path d='M12 9v4M12 17h.01'/></svg>");}
.i-check{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M22 11.08V12a10 10 0 1 1-5.93-9.14'/><path d='M22 4L12 14.01l-3-3'/></svg>");}
.i-msg{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linejoin='round'><path d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'/></svg>");}
.i-book{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linejoin='round'><path d='M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z'/><path d='M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z'/></svg>");}
.i-net{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2'><circle cx='18' cy='5' r='3'/><circle cx='6' cy='12' r='3'/><circle cx='18' cy='19' r='3'/><path d='M8.59 13.51l6.83 3.98M15.41 6.51l-6.82 3.98'/></svg>");}
.i-chart{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'><path d='M12 20V10M18 20V4M6 20v-4'/></svg>");}
.i-info{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'><circle cx='12' cy='12' r='10'/><path d='M12 16v-4M12 8h.01'/></svg>");}
.i-code{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M16 18l6-6-6-6M8 6l-6 6 6 6'/></svg>");}
.i-doc{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linejoin='round'><path d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/><path d='M14 2v6h6M16 13H8M16 17H8'/></svg>");}
.i-layers{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linejoin='round'><path d='M12 2L2 7l10 5 10-5-10-5z'/><path d='M2 17l10 5 10-5M2 12l10 5 10-5'/></svg>");}
.i-shield{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linejoin='round'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/></svg>");}
.i-send{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linejoin='round'><path d='M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z'/></svg>");}
.i-type{--i:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'><path d='M4 7V4h16v3M9 20h6M12 4v16'/></svg>");}

/* typographic hierarchy */
.stApp h1{ font-size:clamp(34px,6vw,58px); font-weight:900; letter-spacing:-1.5px; line-height:1;
    color:var(--heading) !important; margin:0; }
h2.seclabel{ display:flex; align-items:center; gap:10px; font-size:12px; font-weight:800;
    letter-spacing:3px; text-transform:uppercase; color:var(--slate);
    margin:var(--s6) 0 var(--s4); border:none; padding:0; }
h2.seclabel .ico{ width:14px; height:14px; opacity:.9; }
h2.seclabel::after{ content:""; flex:1; height:1px;
    background:linear-gradient(90deg, var(--line), transparent 85%); }
h3{ color:var(--heading); margin:0; }
.body{ font-size:17px; color:var(--mute); line-height:1.6; }
.small{ font-size:12.5px; color:var(--mute); line-height:1.6; }
.label{ font-size:11px; font-weight:800; letter-spacing:2.5px; text-transform:uppercase; }

/* unconventional divider: three fireflies */
.dots{ display:flex; justify-content:center; gap:10px; margin:var(--s5) 0 0; }
.dots span{ width:5px; height:5px; border-radius:50%; background:var(--slate); opacity:.35; }
.dots span:nth-child(2){ background:var(--coral); opacity:.55;
    box-shadow:0 0 8px rgba(238,108,77,.5); }

/* accessibility */
html{ scroll-behavior:smooth; }
button:focus-visible, a:focus-visible, textarea:focus-visible, [role="button"]:focus-visible{
  outline:3px solid var(--primary) !important; outline-offset:2px; }
.skiplink{ position:absolute; left:-9999px; top:0; z-index:100; background:var(--slate-deep);
  color:#fff; padding:var(--s2) var(--s3); font-weight:800; text-decoration:none;
  border-radius:0 0 var(--r-input) 0; }
.skiplink:focus{ left:var(--s3); }

/* animations */
@keyframes fadeUp { from{ opacity:0; transform:translateY(18px); } to{ opacity:1; transform:none; } }
@keyframes revealFallback { to{ opacity:1; transform:none; } }
@keyframes glowIn { from{ opacity:0; filter:brightness(1.5); } to{ opacity:1; filter:none; } }
@keyframes pulse { 0%,100%{ opacity:.35; transform:scale(1); } 50%{ opacity:1; transform:scale(1.25); } }
@media (prefers-reduced-motion: no-preference){
  .anim{ animation:fadeUp .7s cubic-bezier(.2,.7,.2,1) both; }
  .d1{ animation-delay:.05s } .d2{ animation-delay:.14s } .d3{ animation-delay:.23s } .d4{ animation-delay:.32s }
  .reveal{ opacity:0; transform:translateY(30px) scale(.99);
           transition:opacity .5s ease, transform .62s cubic-bezier(.2,.75,.2,1);
           animation:revealFallback .4s ease 2.4s forwards; }
  .reveal.in{ opacity:1; transform:none; animation:none; }
  .steps .step:nth-child(2), .reads .readc:nth-child(2), .nums .numc:nth-child(2){ transition-delay:.08s; }
  .steps .step:nth-child(3), .nums .numc:nth-child(3){ transition-delay:.16s; }
  .steps .step:nth-child(4){ transition-delay:.24s; }
  .verdictwrap{ animation:glowIn .6s ease both; }
  .heroart .fl{ animation:pulse 3.2s ease-in-out infinite; }
  .heroart .fl.f2{ animation-delay:1.1s; } .heroart .fl.f3{ animation-delay:2.2s; }
  .lucload span{ animation:pulse 1.2s ease-in-out infinite; }
  .lucload span:nth-child(2){ animation-delay:.2s } .lucload span:nth-child(3){ animation-delay:.4s }
}

/* header */
.masthead{ display:flex; align-items:center; gap:var(--s3); }
.brandmark{ width:56px; height:56px; background-position:center; background-size:contain;
    background-repeat:no-repeat; flex:none;
    filter:drop-shadow(0 0 10px rgba(238,108,77,.25)); }
.brandtext{ display:flex; flex-direction:column; gap:3px; }
.brandname{ font-family:'Century Gothic','Questrial','Josefin Sans','Futura','Trebuchet MS',sans-serif;
            font-weight:300; font-size:31px; letter-spacing:1.5px; color:var(--heading); line-height:1; }
.brandslogan{ font-size:11px; font-weight:700; letter-spacing:2.5px; text-transform:uppercase; color:var(--coral); }
.headrule{ height:0; border-bottom:1px solid var(--line); margin:var(--s2) 0 var(--s1); }

/* back to top (anchor; no JS) */
.totop{ position:fixed; left:var(--s3); bottom:var(--s3); z-index:60; text-decoration:none; display:inline-flex;
        align-items:center; gap:7px; font-size:11px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase;
        color:var(--heading); background:var(--surface); border:1px solid var(--line);
        border-radius:var(--r-pill); box-shadow:var(--shadow-1); padding:9px 14px;
        transition:box-shadow .2s, border-color .2s; }
.totop:hover{ border-color:var(--coral); box-shadow:0 0 0 3px rgba(238,108,77,.15), var(--shadow-1); }
.totop .ar{ font-size:14px; line-height:1; }

/* metrics table */
.dtable{ background:var(--surface); border:1px solid var(--line); border-radius:var(--r-card);
    box-shadow:var(--shadow-1); overflow-x:auto; margin:0 0 var(--s2); }
.dtable table{ width:100%; border-collapse:collapse; font-size:13.5px; min-width:360px; }
.dtable th, .dtable td{ padding:10px var(--s3); text-align:right; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums; }
.dtable th:first-child, .dtable td:first-child{ text-align:left; }
.dtable td:first-child{ font-weight:800; color:var(--heading); }
.dtable thead th{ font-size:11px; letter-spacing:1px; text-transform:uppercase; color:var(--mute); border-bottom:2px solid var(--line); }
.dtable tbody tr:last-child td{ border-bottom:none; }
.dtable tbody tr.lbwin td{ color:var(--coral); font-weight:800; }
.dtable tbody tr.lbdemo td:first-child{ color:var(--slate); }

/* header controls: EN/PT as one sliding pill pair, theme toggle in the same family */
.st-key-lang_en button, .st-key-lang_pt button{
    box-shadow:none !important; text-transform:uppercase; letter-spacing:1px;
    font-weight:800 !important; padding:6px 0 !important; min-height:0 !important;
    border:1px solid var(--line) !important; }
.st-key-lang_en button[kind="secondary"], .st-key-lang_pt button[kind="secondary"]{
    background:var(--surface) !important; color:var(--heading) !important; }
.st-key-lang_en button{ border-radius:var(--r-pill) 0 0 var(--r-pill) !important; }
.st-key-lang_pt button{ border-radius:0 var(--r-pill) var(--r-pill) 0 !important; margin-left:-1px; }
.st-key-lang_en button[kind="primary"], .st-key-lang_pt button[kind="primary"]{
    background:var(--slate-deep) !important; border-color:var(--slate-deep) !important;
    box-shadow:inset 0 0 12px rgba(238,108,77,.35) !important; }
.st-key-dark_toggle button{ box-shadow:none !important; border:1px solid var(--line) !important;
    border-radius:var(--r-pill) !important;
    background:var(--surface) !important; color:var(--heading) !important; font-size:13px !important;
    font-weight:800 !important; padding:6px 4px !important; min-height:0 !important; }

/* hero: two columns, message left + luciola art right */
.land{ margin:2px 0 var(--s2); }
.heroband{ display:grid; grid-template-columns:1.15fr .85fr; gap:var(--s4); align-items:center; }
@media (max-width:760px){ .heroband{ grid-template-columns:1fr; } }
.land .brow{ font-size:12px; font-weight:800; letter-spacing:3px; text-transform:uppercase; color:var(--coral); }
.land h1{ margin:12px 0 0; max-width:16ch; text-wrap:balance; }
.beta{ display:inline-block; vertical-align:top; margin-left:14px; font-size:13px; font-weight:900;
       letter-spacing:2px; text-transform:uppercase; color:#fff; background:var(--coral);
       border-radius:var(--r-pill); padding:4px 12px; box-shadow:0 0 16px rgba(238,108,77,.35); }
.land .tag{ font-size:17px; color:var(--mute); font-weight:400; max-width:52ch; margin:var(--s3) 0 0; line-height:1.6; }
.land .tick{ width:64px; height:5px; background:var(--coral); border-radius:var(--r-pill); margin:var(--s4) 0 0; }

/* luciola art: constellation panel, EN and PT as two connected lights */
.heroart{ position:relative; min-height:240px; border-radius:var(--r-card);
    background:
      radial-gradient(140px 100px at 30% 35%, rgba(238,108,77,.10), transparent 70%),
      radial-gradient(160px 120px at 72% 62%, rgba(61,90,128,.12), transparent 70%),
      var(--surface-2);
    border:1px solid var(--line); box-shadow:var(--shadow-1); overflow:hidden; }
.heroart svg{ position:absolute; inset:0; width:100%; height:100%; }
.heroart .node{ position:absolute; display:flex; align-items:center; justify-content:center;
    width:52px; height:52px; border-radius:50%; font-size:13px; font-weight:900;
    letter-spacing:1px; color:#fff; background:var(--slate-deep);
    box-shadow:0 0 0 5px rgba(61,90,128,.14), 0 0 22px rgba(238,108,77,.28); }
.heroart .node.en{ left:24%; top:28%; }
.heroart .node.pt{ left:60%; top:54%; background:var(--coral);
    box-shadow:0 0 0 5px rgba(238,108,77,.16), 0 0 22px rgba(238,108,77,.4); }
.heroart .fl{ position:absolute; width:6px; height:6px; border-radius:50%; background:var(--coral);
    box-shadow:0 0 10px rgba(238,108,77,.8); opacity:.6; }
.heroart .fl.f1{ left:12%; top:66%; } .heroart .fl.f2{ left:80%; top:22%; }
.heroart .fl.f3{ left:46%; top:12%; width:4px; height:4px; }

/* the finding: elevated accent surface (kept as the one brutal signature) */
.hero{ margin:var(--s5) 0 0; background:linear-gradient(160deg,#243a63,#1F3050 60%,#16223d); color:#fff;
       border:1px solid #16223d; border-radius:var(--r-card); box-shadow:10px 10px 0 var(--coral);
       padding:var(--s4) 28px; display:flex; align-items:center; gap:var(--s4); flex-wrap:wrap; }
.hero .klabel{ font-size:11px; font-weight:800; letter-spacing:2.5px; text-transform:uppercase; color:var(--amber); }
.hero .kmain{ font-size:26px; font-weight:900; letter-spacing:-.5px; margin:6px 0 0; line-height:1.1; }
.hero .kmain b{ color:var(--coral); }
.hero .ksub{ font-size:14px; color:#B9C6DE; margin:var(--s2) 0 0; max-width:46ch; }
.hero .knum{ margin-left:auto; text-align:right; }
.hero .knum .big{ font-size:60px; font-weight:900; letter-spacing:-3px; line-height:.9; font-variant-numeric:tabular-nums;
                  background:linear-gradient(90deg,#F4A261,#EE6C4D); -webkit-background-clip:text;
                  background-clip:text; -webkit-text-fill-color:transparent; }
.hero .knum .cap{ font-size:11px; color:#9DB0D2; font-weight:700; letter-spacing:1px; text-transform:uppercase; }

/* method steps */
.steps{ display:grid; grid-template-columns:repeat(4,1fr); gap:var(--s3); }
@media (max-width:760px){ .steps{ grid-template-columns:repeat(2,1fr); } }
@media (max-width:460px){ .steps{ grid-template-columns:1fr; } }
.step{ background:var(--surface); border:1px solid var(--line); border-radius:var(--r-card);
    box-shadow:var(--shadow-1); padding:var(--s3) var(--s3) 18px;
    transition:transform .2s, box-shadow .2s; }
.step:hover{ transform:translateY(-2px); box-shadow:var(--shadow-2); }
.step .n{ display:inline-flex; align-items:center; gap:7px; font-size:11px; font-weight:900;
    color:var(--slate); background:var(--surface-2); border-radius:var(--r-pill);
    padding:4px 11px; letter-spacing:1px; }
.step .n .ico{ width:12px; height:12px; }
.step h3{ font-size:16px; font-weight:900; margin:12px 0 6px; letter-spacing:-.2px; }
.step p{ font-size:13px; color:var(--mute); line-height:1.55; margin:0; }
.step.coral .n{ background:rgba(238,108,77,.12); color:var(--danger); }
.step.coral h3{ color:var(--coral); }
.step.ship{ grid-column:1 / -1; display:flex; align-items:center; gap:var(--s4); padding:18px 20px; }
.step.ship .shiphead{ display:flex; align-items:center; gap:12px; flex:none; }
.step.ship h3{ margin:0; font-size:19px; }
.step.ship p{ font-size:14px; }
@media (max-width:460px){ .step.ship{ flex-direction:column; align-items:flex-start; gap:10px; } }

/* how it reads text */
.reads{ display:grid; grid-template-columns:1fr 1fr; gap:var(--s3); }
@media (max-width:620px){ .reads{ grid-template-columns:1fr; } }
.readc{ background:var(--surface); border:1px solid var(--line); border-radius:var(--r-card);
    box-shadow:var(--shadow-1); padding:22px; transition:transform .2s, box-shadow .2s; }
.readc:hover{ transform:translateY(-2px); box-shadow:var(--shadow-2); }
.readc.alt{ border-top:3px solid var(--coral); }
.readc .rlab{ font-size:11px; font-weight:800; letter-spacing:2px; text-transform:uppercase; color:var(--slate); }
.readc.alt .rlab{ color:var(--coral); }
.readc h3{ font-size:23px; font-weight:900; margin:5px 0 0; letter-spacing:-.5px; }
.readc p{ font-size:13.5px; color:var(--ink); line-height:1.55; margin:12px 0 14px; }
.readc .ms{ display:flex; gap:7px; flex-wrap:wrap; }
.readc .ms span{ font-size:12px; font-weight:700; color:var(--heading); background:var(--surface-2);
    border-radius:var(--r-pill); padding:5px 12px; }
.readc.alt .ms span{ color:var(--danger); background:rgba(238,108,77,.10); }

/* tech badges */
.techs{ display:flex; gap:var(--s2); flex-wrap:wrap; align-items:center; margin:var(--s4) 0 0; }
.techs .tl{ font-size:11px; font-weight:800; letter-spacing:2px; text-transform:uppercase; color:var(--mute); margin-right:6px; }
.techs span.badge{ display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:700;
    color:var(--slate); background:var(--surface); border:1px solid var(--line);
    border-radius:var(--r-pill); padding:5px 12px; }
.techs span.badge .ico{ width:12px; height:12px; }

/* results stats */
.nums{ display:grid; grid-template-columns:repeat(3,1fr); gap:var(--s3); }
@media (max-width:620px){ .nums{ grid-template-columns:1fr; } }
.numc{ background:linear-gradient(160deg,#243a63,#1F3050 70%); color:#fff; border-radius:var(--r-card);
    border:1px solid #16223d; box-shadow:var(--shadow-1); padding:20px 22px; }
.numc .v{ font-size:40px; font-weight:900; letter-spacing:-2px; line-height:1; font-variant-numeric:tabular-nums; }
.numc .k{ font-size:12.5px; color:#B9C6DE; font-weight:700; margin-top:var(--s2); line-height:1.4; }
.numc.alt{ background:var(--surface); color:var(--heading); border:1px solid var(--line); }
.numc.alt .k{ color:var(--mute); }

/* links */
.links{ display:grid; grid-template-columns:repeat(3,1fr); gap:var(--s3); margin:var(--s4) 0 0; }
@media (max-width:460px){ .links{ grid-template-columns:1fr; } }
.links a{ display:flex; align-items:center; justify-content:center; gap:9px; text-decoration:none;
          font-weight:800; font-size:13px; letter-spacing:.5px; text-transform:uppercase;
          padding:13px 20px; border:1px solid var(--line); border-radius:var(--r-pill);
          color:var(--heading); background:var(--surface); transition:box-shadow .2s, border-color .2s; }
.links a:hover{ border-color:var(--slate); box-shadow:var(--shadow-1); }
.links a.primary{ background:var(--coral); border-color:var(--coral); color:#fff;
          box-shadow:0 0 18px rgba(238,108,77,.35); }
.links a.primary:hover{ background:var(--slate-deep); border-color:var(--slate-deep); }

.anote{ font-size:12.5px; color:var(--mute); line-height:1.6; margin:var(--s4) 0 2px; max-width:82ch; }

/* pipeline infographic */
.pipe{ display:flex; align-items:stretch; gap:0; margin:var(--s4) 0 0; flex-wrap:nowrap; }
.pipe .pnode{ flex:1; display:flex; flex-direction:column; align-items:center; gap:8px;
    background:var(--surface); border:1px solid var(--line); border-radius:var(--r-card);
    padding:14px 8px; text-align:center; min-width:0; }
.pipe .pnode .ico{ width:18px; height:18px; color:var(--slate); }
.pipe .pnode b{ font-size:11px; font-weight:800; letter-spacing:1px; text-transform:uppercase; color:var(--heading); }
.pipe .pnode.lit{ border-color:rgba(238,108,77,.5); box-shadow:0 0 14px rgba(238,108,77,.18); }
.pipe .pnode.lit .ico{ color:var(--coral); }
.pipe .plink{ width:22px; flex:none; align-self:center; height:1px;
    background:linear-gradient(90deg, var(--line), var(--slate)); position:relative; }
.pipe .plink::after{ content:""; position:absolute; right:0; top:-2.5px; width:6px; height:6px;
    border-radius:50%; background:var(--slate); opacity:.6; }
@media (max-width:700px){
  .pipe{ flex-direction:column; align-items:stretch; }
  .pipe .plink{ width:1px; height:18px; align-self:center;
      background:linear-gradient(180deg, var(--line), var(--slate)); }
  .pipe .plink::after{ right:-2.5px; top:auto; bottom:0; }
}

.cta{ margin:var(--s6) 0 var(--s2); text-align:center; }
.cta .go{ display:inline-flex; align-items:center; gap:10px; font-size:13px; font-weight:800;
    letter-spacing:3px; text-transform:uppercase; color:var(--coral); border:none; margin:0; padding:0; }
.cta .go .ico{ width:15px; height:15px; }

/* ------------------------------------------------------------------ tool section */
.panel{ background:linear-gradient(160deg,#243a63,#1F3050 60%,#16223d); color:#fff;
        border:1px solid #16223d; border-radius:var(--r-card); box-shadow:var(--shadow-2);
        padding:var(--s4) 26px; min-height:400px; display:flex; flex-direction:column; }
.panel .brow{ font-size:11px; font-weight:800; letter-spacing:3px; text-transform:uppercase; color:var(--amber); }
.panel .ttl{ font-size:19px; font-weight:800; letter-spacing:-.3px; line-height:1.15; margin:var(--s2) 0 var(--s3); max-width:18ch; }
.panel .hint{ font-size:14px; color:#B9C6DE; font-weight:400; margin-top:auto; line-height:1.6; }
.panel.idle .hint{ margin-top:auto; }

/* one-turn chat: the analyzed text as a message, the verdict as Luciola's reply */
.bubble{ align-self:flex-end; max-width:92%; background:rgba(255,255,255,.09);
    border:1px solid rgba(255,255,255,.12); border-radius:14px 14px 4px 14px;
    padding:10px 14px; font-size:13.5px; line-height:1.5; color:#DCE5F2; }
.bubble .bl{ display:block; font-size:10px; font-weight:800; letter-spacing:1.5px;
    text-transform:uppercase; color:#9DB0D2; margin-bottom:3px; }
.replyhead{ display:flex; align-items:center; gap:8px; margin:var(--s3) 0 var(--s2);
    font-size:10px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; color:#9DB0D2; }
.replyhead .ico{ width:13px; height:13px; color:var(--amber); }

/* verdict composition: seal + number + bars */
.verdictwrap{ display:flex; flex-direction:column; }
.verdict{ display:inline-flex; align-items:center; gap:9px; align-self:flex-start;
    font-size:14px; font-weight:900; letter-spacing:2px; text-transform:uppercase;
    color:#fff; background:rgba(255,255,255,.06); border:1px solid var(--vc);
    border-radius:var(--r-pill); padding:7px 16px;
    box-shadow:0 0 18px color-mix(in srgb, var(--vc) 45%, transparent); }
.verdict .ico{ width:15px; height:15px; color:var(--vc); }
.verdict .vword{ color:var(--vc); }
.verdict .near{ font-size:10px; font-weight:800; letter-spacing:1px; color:#F4B266;
    border-left:1px solid rgba(255,255,255,.25); padding-left:9px; text-transform:none; }
.panel .giant{ font-size:84px; font-weight:900; letter-spacing:-4px; line-height:.9;
    margin:var(--s3) 0 0; font-variant-numeric:tabular-nums; }
.panel .giant .u{ font-size:30px; font-weight:700; letter-spacing:-1px; color:#93A2C0; }
.panel .meter{ position:relative; height:8px; border-radius:var(--r-pill);
    background:rgba(255,255,255,.14); margin:16px 0 6px; overflow:visible; }
.panel .meter .fill{ height:100%; border-radius:var(--r-pill);
    background:linear-gradient(90deg, color-mix(in srgb, var(--vc) 55%, transparent), var(--vc)); }
.panel .meter .thr{ position:absolute; top:-4px; bottom:-4px; width:2px; background:#fff; opacity:.85; }
.panel .meta{ font-size:11.5px; color:#9DB0D2; font-weight:600; letter-spacing:.3px; }
/* class distribution: dual segmented bar */
.dist{ margin:var(--s3) 0 0; }
.dist .dl{ font-size:10px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase; color:#9DB0D2; margin-bottom:6px; }
.dist .dbar{ display:flex; height:14px; border-radius:var(--r-pill); overflow:hidden; }
.dist .dbar i{ display:block; height:100%; }
.dist .dlegend{ display:flex; justify-content:space-between; font-size:11px; color:#B9C6DE; margin-top:5px; font-variant-numeric:tabular-nums; }
.panel .status{ display:flex; align-items:center; gap:7px; font-size:12px; font-weight:800;
    letter-spacing:1px; text-transform:uppercase; color:#7FE0B0; margin-top:var(--s3); }
.panel .status .ico{ width:13px; height:13px; }
.panel.err .status{ color:#FFA98C; }
.lucload{ display:inline-flex; gap:6px; margin-top:var(--s3); }
.lucload span{ width:7px; height:7px; border-radius:50%; background:var(--amber);
    box-shadow:0 0 8px rgba(244,162,97,.7); opacity:.4; }

/* input card */
.st-key-input_card{ background:var(--surface); border:1px solid var(--line);
    border-radius:var(--r-card); box-shadow:var(--shadow-1); padding:var(--s4); }
.st-key-input_card .cardlabel{ display:flex; align-items:center; gap:8px; font-size:11px;
    font-weight:800; letter-spacing:2.5px; text-transform:uppercase; color:var(--mute); margin:0 0 12px; }
.st-key-input_card .cardlabel .ico{ width:13px; height:13px; color:var(--slate); }
.st-key-input_card .exlabel{ font-size:11px; font-weight:800; letter-spacing:1.5px;
    text-transform:uppercase; color:var(--slate); margin:0 0 6px; }
@media (min-width: 768px){
  [data-testid="stColumn"]:has(.st-key-input_card){ order: 2; }
}
.st-key-input_card .stButton button[kind="secondary"]{
    border-radius:var(--r-pill); border:1px solid var(--line); background:var(--surface);
    color:var(--heading); font-weight:700; font-size:12.5px; padding:6px 10px; box-shadow:none;
    transition:border-color .2s, box-shadow .2s; }
.st-key-input_card .stButton button[kind="secondary"]:hover{
    border-color:var(--coral); color:var(--danger);
    box-shadow:0 0 0 3px rgba(238,108,77,.12); background:var(--surface); }
.stButton button[kind="primary"]{
    border-radius:var(--r-pill); border:none; background:var(--slate-deep); color:#fff;
    font-weight:900; letter-spacing:1px; text-transform:uppercase;
    box-shadow:0 0 16px rgba(238,108,77,.25); padding:12px 24px;
    transition:background .2s, box-shadow .2s; }
.stButton button[kind="primary"]:hover{ background:var(--coral);
    box-shadow:0 0 22px rgba(238,108,77,.45); }
.stTextArea textarea{ border-radius:var(--r-input) !important; border:1px solid var(--line) !important;
    background:var(--surface) !important; color:var(--ink) !important;
    font-family:'Lato',sans-serif !important; font-size:15px !important; }
.stTextArea textarea:focus{ border-color:var(--coral) !important;
    box-shadow:0 0 0 3px rgba(238,108,77,.15) !important; }
.stSpinner > div{ border-top-color:var(--coral) !important; }

/* explainability: methodological surface, distinct from the verdict */
.explain{ background:var(--surface-2); border:1px solid var(--line); border-radius:var(--r-card);
          box-shadow:var(--shadow-1); padding:var(--s4); margin-top:var(--s4); }
.explain h3{ display:flex; align-items:center; gap:9px; font-size:12px; font-weight:800;
    letter-spacing:2.5px; text-transform:uppercase; color:var(--slate); margin:0 0 var(--s3); }
.explain h3 .ico{ width:14px; height:14px; }
.explain .facts{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
                 gap:var(--s2) var(--s4); margin:0 0 var(--s3); }
.explain .fact .fk{ display:flex; align-items:center; gap:6px; font-size:10.5px; font-weight:800;
    letter-spacing:1.5px; text-transform:uppercase; color:var(--mute); }
.explain .fact .fk .ico{ width:11px; height:11px; }
.explain .fact .fv{ font-size:14px; font-weight:700; color:var(--heading); font-variant-numeric:tabular-nums; margin-top:2px; }
.explain .fact .cdots{ display:inline-flex; gap:4px; margin-left:7px; vertical-align:1px; }
.explain .fact .cdots i{ width:7px; height:7px; border-radius:50%; background:var(--line); }
.explain .fact .cdots i.on{ background:var(--slate); box-shadow:0 0 6px rgba(61,90,128,.5); }
.explain .termlab{ font-size:10.5px; font-weight:800; letter-spacing:1.5px; text-transform:uppercase;
                   color:var(--mute); margin:var(--s3) 0 var(--s2); }
.explain .terms{ display:flex; gap:var(--s2); flex-wrap:wrap; }
.explain .term{ position:relative; font-family:'SFMono-Regular',Consolas,monospace; font-size:13px;
    font-weight:700; white-space:pre; padding:6px 11px 9px; background:var(--surface);
    border:1px solid var(--line); border-radius:8px; overflow:hidden; }
.explain .term::after{ content:""; position:absolute; left:0; bottom:0; height:3px;
    width:var(--w,50%); background:currentColor; opacity:.55; }
.explain .term.up{ color:var(--danger); border-color:color-mix(in srgb, var(--danger) 45%, var(--line)); }
.explain .term.down{ color:var(--success); border-color:color-mix(in srgb, var(--success) 45%, var(--line)); }
.explain .legend{ font-size:11.5px; color:var(--mute); margin-top:var(--s2); }
.explain .legend .sw{ display:inline-block; width:9px; height:9px; border-radius:50%;
    vertical-align:middle; margin:0 5px 2px 12px; }
.explain .note{ font-size:12.5px; color:var(--mute); line-height:1.6; margin:var(--s3) 0 0; max-width:82ch; }

/* footer */
.sitefoot{ display:flex; align-items:flex-start; gap:12px; background:var(--surface);
    border:1px solid var(--line); border-left:4px solid var(--coral);
    border-radius:var(--r-card); box-shadow:var(--shadow-1);
    padding:var(--s3) 18px; font-size:13.5px; color:var(--mute); margin-top:var(--s4); line-height:1.6; }
.sitefoot .ico{ width:16px; height:16px; color:var(--coral); flex:none; margin-top:3px; }
.sitefoot b{ color:var(--heading); } .sitefoot a{ color:var(--slate); font-weight:700; }
</style>
""",
    unsafe_allow_html=True,
)

st.session_state.setdefault("dark", False)
if st.session_state.dark:
    st.markdown(DARK_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- state
st.session_state.setdefault("lang", "en")
st.session_state.setdefault("text", "")
st.session_state.setdefault("classified", False)


def set_lang(value: str):
    st.session_state.lang = value


def toggle_dark():
    st.session_state.dark = not st.session_state.dark


def use_example(value: str):
    st.session_state.text = value
    st.session_state.classified = True


def classify_now():
    st.session_state.classified = True


# --------------------------------------------------------------------------- header
lang = st.session_state.lang
t = T[lang]

st.markdown(
    f'<a class="skiplink" href="#tool">{t["skip"]}</a>'
    f'<div id="top"></div>'
    f'<a href="#top" class="totop" aria-label="{t["totop"]}"><span class="ar">&#8593;</span> {t["totop"]}</a>',
    unsafe_allow_html=True,
)

brand, controls = st.columns([0.58, 0.42], vertical_alignment="center")
with brand:
    st.markdown(
        f'<style>.brandmark{{background-image:url("{MARK_URI}")}}</style>'
        '<header class="masthead"><div class="brandmark" role="img" aria-label="Luciola"></div>'
        '<div class="brandtext"><div class="brandname">Luciola</div>'
        '<div class="brandslogan">many small lights</div></div></header>',
        unsafe_allow_html=True,
    )
with controls:
    cc = st.columns([1, 1, 1.3], gap="small")
    cc[0].button("EN", key="lang_en", on_click=set_lang, args=("en",), use_container_width=True,
                 type="primary" if lang == "en" else "secondary")
    cc[1].button("PT", key="lang_pt", on_click=set_lang, args=("pt",), use_container_width=True,
                 type="primary" if lang == "pt" else "secondary")
    cc[2].button(t["theme_light"] if st.session_state.dark else t["theme_dark"],
                 key="dark_toggle", on_click=toggle_dark, use_container_width=True)

st.markdown('<div class="headrule" role="presentation"></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- landing
_step_icons = ["i-layers", "i-shield", "i-cpu", "i-chart", "i-send"]
_steps_html = ""
for i, (title, body) in enumerate(t["steps"]):
    icon = ico(_step_icons[i])
    if i == 4:
        _steps_html += (
            f'<article class="step ship coral reveal"><div class="shiphead">'
            f'<span class="n">{icon} 05</span>'
            f'<h3>{title}</h3></div><p>{body}</p></article>'
        )
    else:
        _steps_html += (
            f'<article class="step reveal"><span class="n">{icon} 0{i + 1}</span>'
            f'<h3>{title}</h3><p>{body}</p></article>'
        )

_r1 = "".join(f"<span>{m}</span>" for m in t["read1_models"])
_r2 = "".join(f"<span>{m}</span>" for m in t["read2_models"])
_techs = "".join(f'<span class="badge">{ico("i-net")} {x}</span>' for x in t["techs"])

# pipeline infographic: TEXT -> LANGUAGE -> MODEL -> VERDICT -> CONFIDENCE -> EXPLANATION
_pipe_html = ""
for i, (plabel, picon) in enumerate(t["pipe"]):
    lit = " lit" if i == 3 else ""
    if i:
        _pipe_html += '<span class="plink" role="presentation"></span>'
    _pipe_html += f'<span class="pnode{lit}">{ico(picon)}<b>{plabel}</b></span>'

# leaderboard table (best result per model, macro-F1, both policies) — backs the Results cards
_dec = (lambda s: s.replace(".", ",")) if lang == "pt" else (lambda s: s)
_lb_data = [
    ("BERTimbau (PT)", "0.784", "0.835", "lbwin"),
    ("twitter-XLM-R", "0.749", "0.766", ""),
    ("XLM-R multilingual", "0.743", "0.764", ""),
    ("BERTweet (EN)", "0.708", "0.753", ""),
    (t["lb_demo"], "0.729", "0.746", "lbdemo"),
]
_lb_rows = ""
for _name, _a, _b, _cls in _lb_data:
    _c = f' class="{_cls}"' if _cls else ""
    _lb_rows += f"<tr{_c}><td>{_name}</td><td>{_dec(_a)}</td><td>{_dec(_b)}</td></tr>"

st.markdown(
    html_block(f"""
<main class="land">
  <section aria-label="{t["eyebrow"]}">
    <div class="heroband">
      <div>
        <p class="brow anim d1">{t["eyebrow"]}</p>
        <h1 class="anim d2">{t["title"]} <span class="beta">Beta</span></h1>
        <p class="tag anim d3">{t["tag"]}</p>
        <div class="tick anim d4" role="presentation"></div>
      </div>
      <div class="heroart anim d3" role="img" aria-label="EN and PT as two connected lights">
        <svg viewBox="0 0 400 260" preserveAspectRatio="none" aria-hidden="true">
          <g fill="none" stroke="#3D5A80" stroke-opacity=".22">
            <path d="M118 92 L260 154"/>
            <path d="M50 180 L118 92 L180 40 L318 62"/>
            <path d="M260 154 L330 210"/>
          </g>
          <g fill="#3D5A80" fill-opacity=".4">
            <circle cx="180" cy="40" r="3"/><circle cx="318" cy="62" r="2.4"/>
            <circle cx="50" cy="180" r="2.6"/><circle cx="330" cy="210" r="3"/>
          </g>
        </svg>
        <span class="node en">EN</span>
        <span class="node pt">PT</span>
        <span class="fl f1" role="presentation"></span>
        <span class="fl f2" role="presentation"></span>
        <span class="fl f3" role="presentation"></span>
      </div>
    </div>

    <article class="hero anim d4">
      <div>
        <p class="klabel">{t["find_label"]}</p>
        <p class="kmain">{t["find_main"]}</p>
        <p class="ksub">{t["find_sub"]}</p>
      </div>
      <div class="knum"><div class="big">{t["find_num"]}</div><div class="cap">{t["find_cap"]}</div></div>
    </article>
  </section>

  <section aria-label="{t["how"]}">
    <h2 class="seclabel reveal">{ico("i-book")} {t["how"]}</h2>
    <div class="steps">{_steps_html}</div>
  </section>

  <section aria-label="{t["reads_label"]}">
    <h2 class="seclabel reveal">{ico("i-type")} {t["reads_label"]}</h2>
    <div class="reads">
      <article class="readc reveal"><p class="rlab">{t["read1_sub"]}</p><h3>{t["read1_title"]}</h3>
        <p>{t["read1_body"]}</p><div class="ms">{_r1}</div></article>
      <article class="readc alt reveal"><p class="rlab">{t["read2_sub"]}</p><h3>{t["read2_title"]}</h3>
        <p>{t["read2_body"]}</p><div class="ms">{_r2}</div></article>
    </div>
    <div class="techs reveal"><span class="tl">{t["tech_label"]}</span>{_techs}</div>
  </section>

  <section aria-label="{t["results"]}">
    <h2 class="seclabel reveal">{ico("i-chart")} {t["results"]}</h2>
    <div class="nums">
      <article class="numc reveal"><div class="v">{t["num1_v"]}</div><div class="k">{t["num1_k"]}</div></article>
      <article class="numc alt reveal"><div class="v">{t["num2_v"]}</div><div class="k">{t["num2_k"]}</div></article>
      <article class="numc reveal"><div class="v">{t["num3_v"]}</div><div class="k">{t["num3_k"]}</div></article>
    </div>

    <p class="anote reveal" style="margin-bottom:12px">{t["lb_cap"]}</p>
    <div class="dtable reveal">
      <table>
        <thead><tr><th scope="col">{t["lb_model"]}</th><th scope="col">strict</th><th scope="col">broad</th></tr></thead>
        <tbody>{_lb_rows}</tbody>
      </table>
    </div>
  </section>

  <section aria-label="{t["diag_label"]}">
    <h2 class="seclabel reveal">{ico("i-info")} {t["diag_label"]}</h2>
    <p class="tag reveal">{t["diag_body"]}</p>
  </section>

  <section aria-label="{t["abl_label"]}">
    <h2 class="seclabel reveal">{ico("i-search")} {t["abl_label"]}</h2>
    <p class="tag reveal">{t["abl_sub"]}</p>
  </section>
</main>
"""),
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- ablation heatmap (interactive)
render_iframe(heatmap_html(lang, st.session_state.dark), height=620)

st.markdown(
    html_block(f"""
<div class="land">
  <p class="anote reveal">{t["abl_foot"]}</p>

  <nav class="links reveal" aria-label="project links">
    <a class="primary" href="{REPO}" target="_blank">{ico("i-code")} {t["link_code"]}</a>
    <a href="{DOCS}" target="_blank">{ico("i-doc")} {t["link_docs"]}</a>
    <a href="{DEMO_REPO}" target="_blank">{ico("i-code")} {t["link_demo"]}</a>
  </nav>

  <div class="dots reveal" role="presentation"><span></span><span></span><span></span></div>

  <div class="cta reveal" id="tool">
    <h2 class="go">{ico("i-fly")} {t["cta"]}</h2>
    <div class="pipe" aria-label="pipeline">{_pipe_html}</div>
  </div>
</div>
"""),
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- compute
text = st.session_state.text
wants_result = st.session_state.classified
has_text = bool(text.strip())

result = None
latency_ms = None
if wants_result and has_text:
    with st.spinner(t["status_analyzing"]):
        _t0 = time.perf_counter()
        result = clf.predict(text)
        latency_ms = (time.perf_counter() - _t0) * 1000


def _shorten(s: str, n: int = 120) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def result_panel_html(res, tr, error: bool) -> str:
    """The verdict card. States: idle, error (empty input), hate, not hate, uncertain.
    One chat turn: the analyzed text appears as a message, the verdict as the reply."""
    if error:
        return (
            '<article class="panel idle err" role="alert">'
            f'<p class="brow">{tr["panel_brow"]}</p>'
            f'<h3 class="ttl">{tr["panel_title"]}</h3>'
            f'<p class="status">{ico("i-alert")} {tr["status_err"]}</p>'
            f'<p class="hint">{tr["panel_hint"]}</p>'
            "</article>"
        )
    if not res:
        return (
            '<article class="panel idle" role="status" aria-live="polite">'
            f'<p class="brow">{tr["panel_brow"]}</p>'
            f'<h3 class="ttl">{tr["panel_title"]}</h3>'
            f'<div class="lucload" role="presentation"><span></span><span></span><span></span></div>'
            f'<p class="hint">{tr["panel_hint"]}</p>'
            "</article>"
        )
    score = float(res["score"])
    pct = score * 100
    is_hate = res["label"] == "hate"
    uncertain = abs(score - clf.threshold) < UNCERTAIN_MARGIN
    color = AMBER if uncertain else (CORAL if is_hate else GREEN)
    vicon = "i-alert" if is_hate else "i-check"
    state = tr["state_hate"] if is_hate else tr["state_nothate"]
    near = f'<span class="near">{tr["state_uncertain"]}</span>' if uncertain else ""
    lg = res["language"]
    done = tr["status_done"]
    if latency_ms is not None:
        done = f"{done} · {latency_ms:.0f} ms"
    not_pct = 100 - pct
    return (
        '<article class="panel" role="status" aria-live="polite">'
        f'<p class="brow">{tr["panel_brow"]}</p>'
        f'<div class="bubble"><span class="bl">{tr["chat_you"]}</span>{_shorten(res["text"])}</div>'
        f'<div class="replyhead">{ico("i-fly")} {tr["chat_bot"]}</div>'
        f'<div class="verdictwrap" style="--vc:{color}">'
        f'<span class="verdict">{ico(vicon)}<span class="vword">{state}</span>{near}</span>'
        f'<p class="giant">{pct:.0f}<span class="u">%</span></p>'
        f'<div class="meter" role="img" aria-label="{tr["meta_prob"]}: {pct:.0f}%">'
        f'<div class="fill" style="width:{pct:.1f}%"></div>'
        f'<div class="thr" style="left:{THRESHOLD_PCT}%"></div>'
        "</div>"
        f'<p class="meta">{tr["meta_prob"]} · {tr["meta_thr"]} {THRESHOLD_PCT}% · '
        f'{lg["detected"]} ({lg["confidence"]}) · {res["model_version"].replace("_s42", "")}</p>'
        f'<div class="dist"><div class="dl">{tr["dist_label"]}</div>'
        f'<div class="dbar"><i style="width:{pct:.1f}%;background:{CORAL}"></i>'
        f'<i style="width:{not_pct:.1f}%;background:{GREEN}"></i></div>'
        f'<div class="dlegend"><span>{tr["state_hate"].lower()} {pct:.0f}%</span>'
        f'<span>{tr["state_nothate"].lower()} {not_pct:.0f}%</span></div></div>'
        f'<p class="status">{ico("i-check")} {done}</p>'
        "</div></article>"
    )


def explain_html(res, tr) -> str:
    """Explainability card: model facts plus the exact linear term attribution."""
    score = float(res["score"])
    margin = abs(score - clf.threshold)
    if margin >= 0.15:
        conf, conf_n = tr["conf_high"], 3
    elif margin >= UNCERTAIN_MARGIN:
        conf, conf_n = tr["conf_med"], 2
    else:
        conf, conf_n = tr["conf_low"], 1
    cdots = "".join(f'<i class="{"on" if i < conf_n else ""}"></i>' for i in range(3))
    lg = res["language"]
    ms = f"{latency_ms:.0f} ms" if latency_ms is not None else "n/d"
    facts = [
        ("i-cpu", tr["exp_model"], tr["exp_model_v"]),
        ("i-zap", tr["exp_latency"], ms),
        ("i-globe", tr["exp_lang"], f'{lg["detected"]} ({lg["confidence"]})'),
        ("i-target", tr["exp_thr"], f"{THRESHOLD_PCT}%"),
        ("i-gauge", tr["exp_conf"], f'{conf}<span class="cdots">{cdots}</span>'),
    ]
    facts_html = "".join(
        f'<div class="fact"><div class="fk">{ico(ic)} {k}</div><div class="fv">{v}</div></div>'
        for ic, k, v in facts
    )
    terms = top_terms(res["text"])
    if terms:
        cmax = max(abs(c) for _, c in terms) or 1.0
        chips = "".join(
            f'<span class="term {"up" if c > 0 else "down"}" '
            f'style="--w:{max(18, round(abs(c) / cmax * 100))}%">{name}</span>'
            for name, c in terms
        )
        terms_html = (
            f'<p class="termlab">{tr["exp_terms"]}</p>'
            f'<div class="terms">{chips}</div>'
            f'<p class="legend"><span class="sw" style="background:var(--danger)"></span>'
            f'{tr["exp_toward"]}<span class="sw" style="background:var(--success)"></span>'
            f'{tr["exp_away"]}</p>'
        )
    else:
        terms_html = f'<p class="termlab">{tr["exp_none"]}</p>'
    return (
        '<section class="explain reveal in" aria-label="explainability">'
        f"<h3>{ico('i-search')} {tr['exp_label']}</h3>"
        f'<div class="facts">{facts_html}</div>'
        f"{terms_html}"
        f'<p class="note">{tr["exp_note"]}</p>'
        "</section>"
    )


# --------------------------------------------------------------------------- tool
# Input first in the DOM so mobile stacks Input then Result then Explanation. On desktop
# a CSS `order` rule moves the result panel to the left for the split-canvas layout.
col_input, col_result = st.columns([1.08, 0.92], gap="large")

with col_input:
    with st.container(key="input_card"):
        st.markdown(f'<h3 class="cardlabel">{ico("i-msg")} {t["your_text"]}</h3>',
                    unsafe_allow_html=True)
        st.markdown(f'<p class="exlabel">{t["ex_label"]}</p>', unsafe_allow_html=True)
        chip_cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            chip_cols[i % 2].button(
                ex[lang], key=f"ex_{ex['key']}", on_click=use_example, args=(ex["text"],),
                use_container_width=True,
            )
        st.text_area(
            t["your_text"], key="text", height=150, label_visibility="collapsed",
            placeholder=t["placeholder"],
        )
        st.button(t["classify"], type="primary", on_click=classify_now, use_container_width=True)

with col_result:
    st.markdown(result_panel_html(result, t, error=wants_result and not has_text),
                unsafe_allow_html=True)

if result is not None:
    st.markdown(explain_html(result, t), unsafe_allow_html=True)

# --------------------------------------------------------------------------- footer
st.markdown(
    f'<footer class="sitefoot" role="contentinfo">{ico("i-info")}<span>{t["disc"]} '
    f'<a href="{REPO}">{t["disc_code"]}</a> · <a href="{DOCS}">{t["disc_docs"]}</a>.</span></footer>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- scroll reveal (cross-browser)
# st.markdown strips <script>, so inject an IntersectionObserver from a 0-height iframe that reaches
# the parent document and adds .in to .reveal elements as they enter view. Works on mobile, Firefox
# and Safari (unlike CSS scroll-timeline). The CSS fallback reveals everything if this is blocked.
render_iframe(
    """
<script>
(function(){
  try{
    var doc = window.parent && window.parent.document;
    if(!doc){ return; }
    var els = doc.querySelectorAll('.reveal:not(.in)');
    var IO = window.parent.IntersectionObserver || window.IntersectionObserver;
    if(!IO){ for(var i=0;i<els.length;i++){ els[i].classList.add('in'); } return; }
    var io = new IO(function(entries){
      for(var j=0;j<entries.length;j++){
        if(entries[j].isIntersecting){ entries[j].target.classList.add('in'); io.unobserve(entries[j].target); }
      }
    }, {root:null, threshold:0.12, rootMargin:'0px 0px -8% 0px'});
    for(var k=0;k<els.length;k++){ io.observe(els[k]); }
  }catch(e){}
})();
</script>
""",
    height=1,
)
