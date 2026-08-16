# Luciola

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

## How it works

Input text is cleaned with the same pipeline used in training, the language is detected
(`lingua`), and TF-IDF features feed the stacked ensemble, whose calibrated score is compared
to a threshold tuned on the validation set (strict policy: only explicit hate is positive). The `hsc` package is vendored
under `src/`; the model bundle and configs ship with the repo, so nothing is fetched at runtime.

## Responsible use

This is not a moderation oracle. It reflects the biases of its training data. The study measures
over-flagging of some identity terms. It should support, never replace, human review. Implicit
hate expressed without slurs is its main blind spot. Predictions are probabilistic, not verdicts.

## License

Code is released under the MIT License. Training-data licenses vary by source and restrict
commercial use; this demo is for research and educational purposes only.
