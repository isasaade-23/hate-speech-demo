# Luciola

**Live at [luciola.65-108-90-215.sslip.io](https://luciola.65-108-90-215.sslip.io).**
That instance runs on our own machine and does not sleep. The Streamlit Community Cloud
deployment is kept as a mirror and hibernates when idle.

**Live demo (EN / PT).** A [Streamlit](https://streamlit.io) demo for a probabilistic hate-speech
classifier over English and Portuguese social-media text. It returns a hate or not-hate label with
a confidence score and the detected language. The interface switches between English and Portuguese
and between a light and dark theme, and it renders the project's stop-word ablation as an
interactive heatmap.

This repository is the deployable demo only. It serves a calibrated stacked ensemble: a meta
logistic regression over three TF-IDF models (Logistic Regression, linear SVM, LightGBM), with
meta weights, decision threshold and Platt calibration all fit on validation only. Trained on
corpus v5 (113,826 rows, eight sources including 41k adversarial synthetic examples). It is
self-contained (32 MB, CPU only, ~30 ms per text). There is also a browser extension that runs
the ensemble's linear member fully in-browser:
[luciola-extension](https://github.com/isasaade-23/luciola-extension).

- Full study, code, and results: https://github.com/isasaade-23/hate-speech-nlp-en-pt
- Documentation: https://isasaade-23.github.io/hate-speech-nlp-en-pt/

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy (Streamlit Community Cloud)

Point [share.streamlit.io](https://share.streamlit.io) at this repository, main file
`streamlit_app.py`. In the app's advanced settings, select **Python 3.11** (the model bundle is
pinned to scikit-learn 1.9.0, which needs Python >= 3.10).

## Deploy (own server)

The instance that backs the public address is a systemd unit running Streamlit on
`127.0.0.1:8501`, behind Caddy, which obtains the certificate on its own. The unit caps memory
(`MemoryMax`) and raises `OOMScoreAdjust`, so a runaway model is killed instead of whatever else
shares the machine. Python 3.14 with scikit-learn 1.9.0 loads the bundle unchanged.

## How it works

Input text is cleaned with the same pipeline used in training, the language is detected
(`lingua`), and TF-IDF features feed the stacked ensemble, whose calibrated score is compared
to a threshold tuned on the validation set (strict policy: only explicit hate is positive). The `hsc` package is vendored
under `src/`; the model bundle and configs ship with the repo, so nothing is fetched at runtime.

## Responsible use

This is not a moderation oracle. It reflects the biases of its training data. An identity-disclosure
probe (2026-08-24) found a Portuguese-only model variant over-flagging real non-hateful identity
sentences by 34.0%, against 13.7% for the model served here; that variant was blocked from release.
The full record is in
[`methodology/pt_recall_x_vies.md`](https://github.com/isasaade-23/hate-speech-nlp-en-pt/blob/main/methodology/pt_recall_x_vies.md).
It should support, never replace, human review. Implicit
hate expressed without slurs is its main blind spot. Predictions are probabilistic, not verdicts.

**It works much worse in Portuguese, and the aggregate score hides it.** On the v5 test the served
ensemble recovers **0.78 of hate in English (n=11,513) and 0.32 in Portuguese (n=4,748)**; the
0.74 average is carried by English. The cause was measured and it is not the method: there is too
little annotated Portuguese data with a licence that allows use. Anyone deploying this on
Portuguese content should assume it misses roughly two thirds of what it is looking for. Numbers
from
[`reports/tables/stack_slices_v5_strict.csv`](https://github.com/isasaade-23/hate-speech-nlp-en-pt/blob/main/reports/tables/stack_slices_v5_strict.csv).

## License

**Code**: [GNU Affero General Public License v3.0](LICENSE). This demo is a network service, so
AGPL section 13 applies: anyone using the hosted app is entitled to the corresponding source,
which is this repository. The footer of the app links here.

**Served model bundle** (`models/stack_strict_s42/`): research and educational use only, no
commercial use. Not under the AGPL, because the training-data licenses restrict what can be
granted downstream. See [`LICENSE-MODEL.md`](LICENSE-MODEL.md).
