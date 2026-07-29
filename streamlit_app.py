"""Streamlit demo — bilingual (EN/PT) hate-speech classifier.

Poster / split-canvas design with an animated landing that scrolls into the live tool.
The interface language (EN/PT) is switchable at the top; the classifier itself handles
both languages regardless.

Serves the CPU product model (tfidf_logreg_strict) directly. Self-contained: the `hsc`
package is vendored under src/, the model bundle and configs travel with the repo.

Research demo. Not a moderation verdict.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from hsc.inference import get_classifier  # noqa: E402

MODEL_ID = "tfidf_logreg_strict_s42"
THRESHOLD_PCT = 66.5  # tuned decision threshold, shown on the meter for transparency

REPO = "https://github.com/isasaade-23/hate-speech-nlp-en-pt"
DOCS = "https://isasaade-23.github.io/hate-speech-nlp-en-pt/"
DEMO_REPO = "https://github.com/isasaade-23/hate-speech-demo"

CORAL = "#EE6C4D"
GREEN = "#1D9E75"

# example texts are content (kept as-is); only the descriptor labels are translated
EXAMPLES = [
    {"key": "en_ok", "en": "EN friendly", "pt": "EN amigável",
     "text": "I love this community, everyone is so welcoming!"},
    {"key": "en_hate", "en": "EN implicit hate", "pt": "EN ódio implícito",
     "text": "those people are subhuman and should be removed"},
    {"key": "pt_ok", "en": "PT friendly", "pt": "PT amigável",
     "text": "que vídeo incrível, parabéns pelo trabalho de vocês"},
    {"key": "pt_hate", "en": "PT hostile", "pt": "PT hostil",
     "text": "vocês são um bando de idiotas e não deviam existir"},
]

# --------------------------------------------------------------------------- i18n
T = {
    "en": {
        "eyebrow": "Research demo · EN / PT",
        "title": "Bilingual Hate-Speech Detection",
        "tag": "A study compares classical models with transformers under one leakage-safe "
               "protocol. Below runs the classifier it produced. Try it.",
        "find_label": "The finding",
        "find_main": "Transformers win. The gap is <b>statistically significant</b>.",
        "find_sub": "XLM-R reaches macro-F1 0.750. The best classical baseline reaches 0.709 "
                    "(paired McNemar, Holm). This demo runs the classical model, 3.6 MB. It stays "
                    "four points behind and needs no GPU.",
        "find_num": "0.750", "find_cap": "best macro-F1",
        "how": "How it was built",
        "steps": [
            ("Harmonize", "Four sources become one binary schema. Tweets and memes in English, "
                          "comments in Portuguese. Labels follow two policies, strict and broad."),
            ("De-leak", "Exact and near-duplicate removal (MinHash/LSH) runs first. The group split "
                        "is frozen. No paraphrase crosses train and test."),
            ("Train", "TF-IDF and SBERT feed LogReg, SVM and LightGBM on CPU. XLM-R, BERTimbau and "
                      "BERTweet are fine-tuned on Colab."),
            ("Evaluate", "Macro-F1, paired McNemar with Holm, calibration (ECE), an identity-term "
                         "bias probe, cross-lingual transfer."),
            ("Ship", "Pareto picks tfidf_logreg. 3.6 MB, 1.6 ms on CPU. It answers you below."),
        ],
        "reads_label": "How it reads text",
        "read1_title": "Word by word", "read1_sub": "bag-of-words",
        "read1_body": "It counts words and character patterns (TF-IDF). It sees no context beyond "
                      "short windows. In return it is fast, 3.6 MB, and runs on any CPU.",
        "read1_models": ["Logistic Regression", "Linear SVM", "LightGBM"],
        "read2_title": "The whole sentence", "read2_sub": "contextual",
        "read2_body": "It reads the full sentence in context, through multilingual embeddings and "
                      "fine-tuned transformers. It costs more. It grasps meaning and transfers "
                      "across languages. The accuracy comes from here.",
        "read2_models": ["multilingual SBERT", "XLM-RoBERTa", "BERTimbau", "BERTweet"],
        "results": "Results",
        "num1_v": "0.750", "num1_k": "best transformer<br>(XLM-R, macro-F1)",
        "num2_v": "0.709", "num2_k": "this demo<br>(classical MVP, macro-F1)",
        "num3_v": "0.42→0.63", "num3_k": "EN→PT zero-shot transfer<br>(TF-IDF → multilingual SBERT)",
        "link_code": "Code &amp; study", "link_docs": "Documentation", "link_demo": "Demo source",
        "cta": "Try it live",
        "panel_brow": "EN / PT · research demo",
        "panel_title": "Bilingual Hate-Speech Classifier",
        "panel_hint": "Type or pick an example on the right. Click Classify. The verdict and its "
                      "probability show here.",
        "state_hate": "Hate", "state_nothate": "Not hate",
        "meta_prob": "hate probability", "meta_thr": "threshold",
        "your_text": "Your text",
        "placeholder": "Type English or Portuguese text...",
        "classify": "Classify",
        "disc": "<b>Responsible use.</b> This is not a moderation oracle. It carries the biases of "
                "its training data; the study measures over-flagging of some identity terms. It "
                "should support human review, not replace it. Implicit hate without slurs is where "
                "it fails most. Research and educational use only.",
        "disc_code": "Code", "disc_docs": "Docs",
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
        "find_label": "O achado",
        "find_main": "Os transformers vencem. A diferença é <b>estatisticamente significativa</b>.",
        "find_sub": "O XLM-R chega a macro-F1 0,750. O melhor clássico chega a 0,709 (McNemar "
                    "pareado, Holm). Este demo roda o clássico, 3,6 MB. Fica quatro pontos atrás e "
                    "dispensa GPU.",
        "find_num": "0,750", "find_cap": "melhor macro-F1",
        "how": "Como foi construído",
        "steps": [
            ("Harmonizar", "Quatro fontes viram um esquema binário. Tweets e memes em inglês, "
                           "comentários em português. Os rótulos seguem duas políticas, strict e broad."),
            ("Anti-vazamento", "Primeiro remove duplicatas exatas e quase-duplicatas (MinHash/LSH). "
                               "O split por grupo é congelado. Nenhuma paráfrase cruza treino e teste."),
            ("Treinar", "TF-IDF e SBERT alimentam LogReg, SVM e LightGBM na CPU. XLM-R, BERTimbau e "
                        "BERTweet são ajustados no Colab."),
            ("Avaliar", "Macro-F1, McNemar pareado com Holm, calibração (ECE), sonda de viés por "
                        "termo de identidade, transferência entre idiomas."),
            ("Publicar", "Pareto escolhe o tfidf_logreg. 3,6 MB, 1,6 ms na CPU. É ele que responde "
                         "abaixo."),
        ],
        "reads_label": "Como ela lê o texto",
        "read1_title": "Palavra por palavra", "read1_sub": "saco de palavras",
        "read1_body": "Conta palavras e padrões de caractere (TF-IDF). Não vê contexto além de "
                      "janelas curtas. Em troca, é rápido, ocupa 3,6 MB e roda em qualquer CPU.",
        "read1_models": ["Logistic Regression", "Linear SVM", "LightGBM"],
        "read2_title": "A frase inteira", "read2_sub": "contextual",
        "read2_body": "Lê a frase inteira em contexto, por embeddings multilíngues e transformers "
                      "ajustados. Custa mais. Capta o sentido e transfere entre idiomas. A acurácia "
                      "vem daí.",
        "read2_models": ["multilingual SBERT", "XLM-RoBERTa", "BERTimbau", "BERTweet"],
        "results": "Resultados",
        "num1_v": "0,750", "num1_k": "melhor transformer<br>(XLM-R, macro-F1)",
        "num2_v": "0,709", "num2_k": "este demo<br>(MVP clássico, macro-F1)",
        "num3_v": "0,42→0,63", "num3_k": "transferência EN→PT zero-shot<br>(TF-IDF → SBERT multilíngue)",
        "link_code": "Código &amp; estudo", "link_docs": "Documentação", "link_demo": "Código do demo",
        "cta": "Experimente ao vivo",
        "panel_brow": "EN / PT · demo de pesquisa",
        "panel_title": "Classificador Bilíngue de Discurso de Ódio",
        "panel_hint": "Digite ou escolha um exemplo à direita. Clique em Classificar. O veredito e a "
                      "probabilidade aparecem aqui.",
        "state_hate": "Ódio", "state_nothate": "Não é ódio",
        "meta_prob": "probabilidade de ódio", "meta_thr": "limiar",
        "your_text": "Seu texto",
        "placeholder": "Digite um texto em inglês ou português...",
        "classify": "Classificar",
        "disc": "<b>Uso responsável.</b> Isto não é um oráculo de moderação. Carrega os vieses dos "
                "dados de treino; o estudo mede a super-marcação de alguns termos de identidade. "
                "Serve para apoiar a revisão humana, não para substituí-la. Falha mais no ódio "
                "implícito, sem palavrão. Uso apenas para pesquisa e educação.",
        "disc_code": "Código", "disc_docs": "Docs",
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
html,body{margin:0;background:transparent;font-family:'Lato',system-ui,sans-serif;color:#1F3050}
.wrap{padding:2px 16px 14px 4px}
.seg{display:inline-flex;border:2px solid #1F3050;margin:0 0 16px}
.seg button{font-family:inherit;font-size:12.5px;font-weight:800;letter-spacing:.6px;text-transform:uppercase;padding:8px 15px;border:none;border-right:2px solid #1F3050;background:#fff;color:#1F3050;cursor:pointer;transition:background .2s,color .2s}
.seg button:last-child{border-right:none}
.seg button[aria-pressed="true"]{background:#1F3050;color:#fff}
.card{background:#fff;border:3px solid #1F3050;box-shadow:8px 8px 0 #3D5A80;padding:14px 16px 12px;max-width:100%;overflow-x:auto}
.grid{display:grid;grid-template-columns:minmax(150px,1.5fr) repeat(3,1fr);gap:6px;min-width:470px}
.hcell{font-size:11px;font-weight:800;letter-spacing:1px;text-transform:uppercase;color:#6B7280;padding:2px 6px 6px;align-self:end;text-align:center}
.hcell.corner{text-align:left}
.rlab{font-size:13.5px;font-weight:900;color:#1F3050;padding:6px 8px;display:flex;flex-direction:column;justify-content:center;transition:color .2s}
.rlab .pol{font-size:10.5px;font-weight:800;letter-spacing:.5px;text-transform:uppercase;color:#3D5A80;margin-top:2px}
.cell{height:46px;display:flex;align-items:center;justify-content:center;font-family:'SFMono-Regular',Consolas,monospace;font-size:13.5px;font-weight:700;font-variant-numeric:tabular-nums;cursor:pointer;border:2px solid transparent;transition:background-color .4s ease,color .4s ease;animation:cellin .45s both}
.cell:hover{border-color:#1F3050}
@keyframes cellin{from{opacity:0;transform:scale(.86)}to{opacity:1;transform:none}}
@media (prefers-reduced-motion:reduce){.cell{animation:none}}
.legend{display:flex;align-items:center;gap:10px;margin:12px 2px 0;font-size:11.5px;font-weight:700;color:#6B7280}
.legbar{height:12px;width:130px;border:1px solid rgba(0,0,0,.15)}
.take{margin:13px 2px 0;font-size:13px;color:#1F3050;line-height:1.5;max-width:66ch}
.take b{color:#EE6C4D}
.tip{position:fixed;pointer-events:none;opacity:0;transform:translate(-50%,-100%);background:#1F3050;color:#fff;border:2px solid #fff;box-shadow:3px 3px 0 #EE6C4D;padding:9px 12px;font-size:12.5px;z-index:30;min-width:180px;transition:opacity .08s}
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
 {model:"LogReg",policy:"strict",f1:[0.7094,0.7139],auc:[0.8413,0.8404],rec:[0.4631,0.5437]},
 {model:"LogReg",policy:"broad",f1:[0.6976,0.6936],auc:[0.7684,0.7678],rec:[0.5475,0.5456]},
 {model:"LightGBM",policy:"strict",f1:[0.7065,0.7012],auc:[0.8200,0.8181],rec:[0.5575,0.5403]},
 {model:"LightGBM",policy:"broad",f1:[0.6983,0.6978],auc:[0.7679,0.7690],rec:[0.5513,0.5206]},
 {model:"SVM",policy:"strict",f1:[0.6716,0.6641],auc:[0.7722,0.7727],rec:[0.4974,0.5214]},
 {model:"SVM",policy:"broad",f1:[0.6734,0.6647],auc:[0.7363,0.7344],rec:[0.5250,0.4675]}
];
const METRICS=[{k:"f1",label:"macro-F1"},{k:"auc",label:"ROC-AUC"},{k:"rec",label:"__REC_LABEL__"}];
const LB={base:"__TT_BASE__",nostop:"__TT_NOSTOP__",delta:"__TT_DELTA__"};
const DELTA_FULL=0.04;
const RANGE={};
METRICS.forEach(function(m){var v=[];DATA.forEach(function(d){v.push(d[m.k][0],d[m.k][1])});RANGE[m.k]=[Math.min.apply(null,v),Math.max.apply(null,v)];});
var mode="base";
function clamp(x,a,b){return Math.max(a,Math.min(b,x))}
function lp(a,b,t){return [Math.round(a[0]+(b[0]-a[0])*t),Math.round(a[1]+(b[1]-a[1])*t),Math.round(a[2]+(b[2]-a[2])*t)]}
function seq(v,k){var r=RANGE[k];var t=r[1]>r[0]?(v-r[0])/(r[1]-r[0]):.5;return lp([233,239,246],[31,48,80],clamp(t,0,1))}
function dv(d){var t=clamp(d/DELTA_FULL,-1,1);return t>=0?lp([246,242,238],[29,158,117],t):lp([246,242,238],[238,108,77],-t)}
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
  if(mode==="delta"){L.innerHTML='<span>__LEG_WORSE__</span><span class="legbar" style="background:linear-gradient(90deg,#EE6C4D,#F6F2EE,#1D9E75)"></span><span>__LEG_BETTER__</span>'}
  else{L.innerHTML='<span>__LEG_LO__</span><span class="legbar" style="background:linear-gradient(90deg,#E9EFF6,#1F3050)"></span><span>__LEG_HI__</span>'}
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


def heatmap_html(language: str) -> str:
    html = HEATMAP_TMPL
    for key, val in HM[language].items():
        html = html.replace("__" + key + "__", val)
    return html


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

st.set_page_config(page_title="Bilingual Hate-Speech Classifier (EN/PT) · beta", layout="wide")

# --------------------------------------------------------------------------- CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&display=swap');

:root{ --slate:#3D5A80; --slate-deep:#1F3050; --coral:#EE6C4D; --amber:#F4A261;
       --green:#1D9E75; --bone:#F6F2EE; --ink:#2B2B2B; --mute:#6B7280; --line:#DAD2C7; }

html, body, [class*="css"], .stApp { font-family:'Lato', sans-serif; }
.stApp { background:var(--bone); }
header[data-testid="stHeader"]{ display:none; }
#MainMenu, footer, [data-testid="stToolbar"]{ display:none; }
.block-container{ padding-top:1rem; padding-bottom:2rem; max-width:1050px; }

/* animations */
@keyframes fadeUp { from{ opacity:0; transform:translateY(18px); } to{ opacity:1; transform:none; } }
@media (prefers-reduced-motion: no-preference){
  .anim{ animation:fadeUp .7s cubic-bezier(.2,.7,.2,1) both; }
  .d1{ animation-delay:.05s } .d2{ animation-delay:.14s } .d3{ animation-delay:.23s } .d4{ animation-delay:.32s }
  @supports (animation-timeline: view()){
    .reveal{ animation:fadeUp both; animation-timeline:view(); animation-range:entry 4% cover 26%; }
  }
}

/* language toggle (compact, no offset shadow) */
.st-key-lang_en button, .st-key-lang_pt button{
    box-shadow:none !important; text-transform:uppercase; letter-spacing:1px;
    font-weight:800 !important; padding:6px 0 !important; min-height:0 !important; }

/* landing */
.land{ margin:2px 0 8px; }
.land .brow{ font-size:12px; font-weight:800; letter-spacing:3px; text-transform:uppercase; color:var(--coral); }
.land h1{ font-size:clamp(34px,6vw,60px); font-weight:900; letter-spacing:-1.5px; line-height:.98;
          color:var(--slate-deep); margin:12px 0 0; max-width:16ch; text-wrap:balance; }
.beta{ display:inline-block; vertical-align:top; margin-left:14px; font-size:15px; font-weight:900;
       letter-spacing:2px; text-transform:uppercase; color:#fff; background:var(--coral);
       border:2px solid var(--slate-deep); box-shadow:3px 3px 0 var(--slate-deep); padding:3px 10px; }
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
.hero .knum .big{ font-size:60px; font-weight:900; letter-spacing:-3px; line-height:.9; font-variant-numeric:tabular-nums;
                  background:linear-gradient(90deg,#F4A261,#EE6C4D); -webkit-background-clip:text;
                  background-clip:text; -webkit-text-fill-color:transparent; }
.hero .knum .cap{ font-size:11px; color:#9DB0D2; font-weight:700; letter-spacing:1px; text-transform:uppercase; }

.seclabel{ font-size:12px; font-weight:800; letter-spacing:3px; text-transform:uppercase; color:var(--slate);
           border-bottom:2px solid var(--line); padding-bottom:8px; margin:46px 0 22px; }

.steps{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
@media (max-width:760px){ .steps{ grid-template-columns:repeat(2,1fr); } }
@media (max-width:460px){ .steps{ grid-template-columns:1fr; } }
.step{ background:#fff; border:2px solid var(--slate-deep); box-shadow:6px 6px 0 var(--slate); padding:16px 16px 18px; }
.step .n{ font-family:'Lato'; font-size:13px; font-weight:900; color:#fff; background:var(--slate-deep);
          display:inline-block; padding:2px 9px; letter-spacing:1px; }
.step h4{ font-size:16px; font-weight:900; color:var(--slate-deep); margin:12px 0 6px; letter-spacing:-.2px; }
.step p{ font-size:13px; color:var(--mute); line-height:1.55; margin:0; }
.step.coral{ box-shadow:6px 6px 0 var(--coral); }
.step.coral .n{ background:var(--coral); }
.step.coral h4{ color:var(--coral); }
.step.ship{ grid-column:1 / -1; display:flex; align-items:center; gap:24px; padding:18px 20px; }
.step.ship .shiphead{ display:flex; align-items:center; gap:12px; flex:none; }
.step.ship h4{ margin:0; font-size:19px; }
.step.ship p{ font-size:14px; }
@media (max-width:460px){ .step.ship{ flex-direction:column; align-items:flex-start; gap:10px; } }

/* how it reads text — two contrasting cards */
.reads{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:620px){ .reads{ grid-template-columns:1fr; } }
.readc{ background:#fff; border:3px solid var(--slate-deep); box-shadow:8px 8px 0 var(--slate); padding:22px; }
.readc.alt{ box-shadow:8px 8px 0 var(--coral); }
.readc .rlab{ font-size:11px; font-weight:800; letter-spacing:2px; text-transform:uppercase; color:var(--slate); }
.readc.alt .rlab{ color:var(--coral); }
.readc h3{ font-size:23px; font-weight:900; color:var(--slate-deep); margin:5px 0 0; letter-spacing:-.5px; }
.readc p{ font-size:13.5px; color:var(--ink); line-height:1.55; margin:12px 0 14px; }
.readc .ms{ display:flex; gap:7px; flex-wrap:wrap; }
.readc .ms span{ font-size:12px; font-weight:800; color:var(--slate-deep); border:2px solid var(--slate-deep); padding:4px 9px; }
.readc.alt .ms span{ border-color:var(--coral); color:var(--coral); }

.nums{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
@media (max-width:620px){ .nums{ grid-template-columns:1fr; } }
.numc{ background:var(--slate-deep); color:#fff; border:3px solid var(--slate-deep); box-shadow:8px 8px 0 var(--coral); padding:20px 22px; }
.numc .v{ font-size:40px; font-weight:900; letter-spacing:-2px; line-height:1; font-variant-numeric:tabular-nums; }
.numc .k{ font-size:12.5px; color:#B9C6DE; font-weight:700; margin-top:8px; line-height:1.4; }
.numc.alt{ background:#fff; color:var(--slate-deep); box-shadow:8px 8px 0 var(--slate); }
.numc.alt .k{ color:var(--mute); }

.links{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:26px 0 0; }
@media (max-width:460px){ .links{ grid-template-columns:1fr; } }
.links a{ text-align:center; text-decoration:none; font-weight:800; font-size:13.5px; letter-spacing:.5px;
          text-transform:uppercase; padding:14px 22px; border:2px solid var(--slate-deep); color:var(--slate-deep); background:#fff; }
.links a:hover{ background:var(--slate-deep); color:#fff; }
.links a.primary{ background:var(--coral); border-color:var(--coral); color:#fff; box-shadow:4px 4px 0 var(--slate-deep); }
.links a.primary:hover{ background:var(--slate-deep); border-color:var(--slate-deep); box-shadow:4px 4px 0 var(--coral); }

.anote{ font-size:12.5px; color:var(--mute); line-height:1.6; margin:22px 0 2px; max-width:82ch; }

.cta{ margin:52px 0 6px; text-align:center; }
.cta .line{ height:2px; background:var(--line); margin-bottom:18px; }
.cta .go{ font-size:13px; font-weight:800; letter-spacing:3px; text-transform:uppercase; color:var(--coral); }
.cta .go .arrow{ display:block; font-size:22px; color:var(--slate); margin-top:4px; }

/* tool */
.panel{ background:linear-gradient(160deg,#243a63,#1F3050 60%,#16223d); color:#fff;
        border:3px solid var(--slate-deep); box-shadow:12px 12px 0 var(--coral);
        padding:30px 28px; min-height:360px; display:flex; flex-direction:column; }
.panel .brow{ font-size:11px; font-weight:800; letter-spacing:3px; text-transform:uppercase; color:var(--amber); }
.panel .ttl{ font-size:20px; font-weight:800; letter-spacing:-.3px; line-height:1.15; margin:8px 0 auto; max-width:16ch; }
.panel .state{ font-size:14px; font-weight:900; letter-spacing:2px; text-transform:uppercase; }
.panel .giant{ font-size:94px; font-weight:900; letter-spacing:-5px; line-height:.85; margin:4px 0 0; font-variant-numeric:tabular-nums; }
.panel .giant .u{ font-size:32px; font-weight:700; letter-spacing:-1px; color:#93A2C0; }
.panel .meter{ position:relative; height:10px; background:rgba(255,255,255,.14); margin:18px 0 8px; }
.panel .meter .fill{ height:100%; }
.panel .meter .thr{ position:absolute; top:-4px; bottom:-4px; width:2px; background:#fff; opacity:.8; }
.panel .meta{ font-size:11.5px; color:#9DB0D2; font-weight:600; letter-spacing:.3px; }
.panel.idle .ttl{ margin-bottom:16px; }
.panel .hint{ font-size:14px; color:#B9C6DE; font-weight:400; margin-top:auto; }

[data-testid="stColumn"]:has([data-testid="stTextArea"]){
    background:#fff; border:3px solid var(--slate-deep); box-shadow:10px 10px 0 var(--slate); padding:26px 24px; }
.rlabel{ font-size:11px; font-weight:800; letter-spacing:2.5px; text-transform:uppercase; color:var(--mute); margin-bottom:12px; }
[data-testid="stColumn"] .stButton button[kind="secondary"]{
    border-radius:0; border:2px solid var(--slate-deep); background:#fff; color:var(--slate-deep);
    font-weight:800; font-size:12.5px; padding:6px 10px; box-shadow:none; }
[data-testid="stColumn"] .stButton button[kind="secondary"]:hover{ background:var(--coral); border-color:var(--coral); color:#fff; }
.stButton button[kind="primary"]{
    border-radius:0; border:2px solid var(--slate-deep); background:var(--slate-deep); color:#fff;
    font-weight:900; letter-spacing:1px; text-transform:uppercase; box-shadow:4px 4px 0 var(--coral); padding:12px 24px; }
.stButton button[kind="primary"]:hover{ background:var(--coral); border-color:var(--coral); box-shadow:4px 4px 0 var(--slate-deep); }
.stTextArea textarea{ border-radius:0 !important; border:2px solid var(--slate-deep) !important;
    background:#fff !important; color:var(--ink) !important; font-family:'Lato',sans-serif !important; font-size:15px !important; }
.disc{ border-left:5px solid var(--coral); background:#fff; border:1px solid var(--line);
       padding:16px 18px; font-size:13.5px; color:var(--mute); margin-top:26px; }
.disc b{ color:var(--slate-deep); } .disc a{ color:var(--slate); font-weight:700; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- state
st.session_state.setdefault("lang", "en")
st.session_state.setdefault("text", "")
st.session_state.setdefault("classified", False)


def set_lang(value: str):
    st.session_state.lang = value


def use_example(value: str):
    st.session_state.text = value
    st.session_state.classified = True


def classify_now():
    st.session_state.classified = True


# --------------------------------------------------------------------------- language toggle
_spacer, toggle = st.columns([0.78, 0.22])
with toggle:
    lc = st.columns(2)
    lc[0].button("EN", key="lang_en", on_click=set_lang, args=("en",), use_container_width=True,
                 type="primary" if st.session_state.lang == "en" else "secondary")
    lc[1].button("PT", key="lang_pt", on_click=set_lang, args=("pt",), use_container_width=True,
                 type="primary" if st.session_state.lang == "pt" else "secondary")

lang = st.session_state.lang
t = T[lang]

# --------------------------------------------------------------------------- landing
_steps_html = ""
for i, (title, body) in enumerate(t["steps"]):
    if i == 4:
        _steps_html += (
            f'<div class="step ship coral reveal"><div class="shiphead"><span class="n">05</span>'
            f'<h4>{title}</h4></div><p>{body}</p></div>'
        )
    else:
        _steps_html += (
            f'<div class="step reveal"><span class="n">0{i + 1}</span><h4>{title}</h4><p>{body}</p></div>'
        )

_r1 = "".join(f"<span>{m}</span>" for m in t["read1_models"])
_r2 = "".join(f"<span>{m}</span>" for m in t["read2_models"])

st.markdown(
    f"""
<div class="land">
  <div class="brow anim d1">{t["eyebrow"]}</div>
  <h1 class="anim d2">{t["title"]} <span class="beta">Beta</span></h1>
  <p class="tag anim d3">{t["tag"]}</p>
  <div class="tick anim d4"></div>

  <div class="hero anim d4">
    <div>
      <div class="klabel">{t["find_label"]}</div>
      <div class="kmain">{t["find_main"]}</div>
      <div class="ksub">{t["find_sub"]}</div>
    </div>
    <div class="knum"><div class="big">{t["find_num"]}</div><div class="cap">{t["find_cap"]}</div></div>
  </div>

  <div class="seclabel reveal">{t["how"]}</div>
  <div class="steps">{_steps_html}</div>

  <div class="seclabel reveal">{t["reads_label"]}</div>
  <div class="reads">
    <div class="readc reveal"><div class="rlab">{t["read1_sub"]}</div><h3>{t["read1_title"]}</h3>
      <p>{t["read1_body"]}</p><div class="ms">{_r1}</div></div>
    <div class="readc alt reveal"><div class="rlab">{t["read2_sub"]}</div><h3>{t["read2_title"]}</h3>
      <p>{t["read2_body"]}</p><div class="ms">{_r2}</div></div>
  </div>

  <div class="seclabel reveal">{t["results"]}</div>
  <div class="nums">
    <div class="numc reveal"><div class="v">{t["num1_v"]}</div><div class="k">{t["num1_k"]}</div></div>
    <div class="numc alt reveal"><div class="v">{t["num2_v"]}</div><div class="k">{t["num2_k"]}</div></div>
    <div class="numc reveal"><div class="v">{t["num3_v"]}</div><div class="k">{t["num3_k"]}</div></div>
  </div>

  <div class="seclabel reveal">{t["abl_label"]}</div>
  <p class="tag reveal">{t["abl_sub"]}</p>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------- ablation heatmap (interactive)
render_iframe(heatmap_html(lang), height=620)

st.markdown(
    f"""
<div class="land">
  <div class="anote reveal">{t["abl_foot"]}</div>

  <div class="links reveal">
    <a class="primary" href="{REPO}" target="_blank">{t["link_code"]}</a>
    <a href="{DOCS}" target="_blank">{t["link_docs"]}</a>
    <a href="{DEMO_REPO}" target="_blank">{t["link_demo"]}</a>
  </div>

  <div class="cta reveal">
    <div class="line"></div>
    <div class="go">{t["cta"]}<span class="arrow">&#8595;</span></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- compute
text = st.session_state.text
show = st.session_state.classified and bool(text.strip())
result = clf.predict(text) if show else None


def left_panel_html(res, tr) -> str:
    if not res:
        return (
            '<div class="panel idle">'
            f'<div class="brow">{tr["panel_brow"]}</div>'
            f'<div class="ttl">{tr["panel_title"]}</div>'
            f'<div class="hint">{tr["panel_hint"]}</div>'
            "</div>"
        )
    score = float(res["score"])
    pct = score * 100
    is_hate = res["label"] == "hate"
    color = CORAL if is_hate else GREEN
    state = tr["state_hate"] if is_hate else tr["state_nothate"]
    lg = res["language"]
    return (
        '<div class="panel">'
        f'<div class="brow">{tr["panel_brow"]}</div>'
        f'<div class="ttl">{tr["panel_title"]}</div>'
        f'<div class="state" style="color:{color}">{state}</div>'
        f'<div class="giant">{pct:.0f}<span class="u">%</span></div>'
        '<div class="meter">'
        f'<div class="fill" style="width:{pct:.1f}%; background:{color}"></div>'
        f'<div class="thr" style="left:{THRESHOLD_PCT}%"></div>'
        "</div>"
        f'<div class="meta">{tr["meta_prob"]} · {tr["meta_thr"]} {THRESHOLD_PCT}% · '
        f'{lg["detected"]} ({lg["confidence"]}) · {res["model_version"].replace("_s42","")}</div>'
        "</div>"
    )


# --------------------------------------------------------------------------- tool
left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.markdown(left_panel_html(result, t), unsafe_allow_html=True)

with right:
    st.markdown(f'<div class="rlabel">{t["your_text"]}</div>', unsafe_allow_html=True)
    chip_cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        chip_cols[i % 2].button(
            ex[lang], key=f"ex_{ex['key']}", on_click=use_example, args=(ex["text"],),
            use_container_width=True,
        )
    st.text_area(
        "Text", key="text", height=150, label_visibility="collapsed", placeholder=t["placeholder"],
    )
    st.button(t["classify"], type="primary", on_click=classify_now, use_container_width=True)

# --------------------------------------------------------------------------- disclaimer
st.markdown(
    f'<div class="disc">{t["disc"]} '
    f'<a href="{REPO}">{t["disc_code"]}</a> · <a href="{DOCS}">{t["disc_docs"]}</a>.</div>',
    unsafe_allow_html=True,
)
