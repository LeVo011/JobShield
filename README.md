# JobShield

**AI-Powered Recruitment Scam Detection and Verification Assistant**

JobShield is an educational, local-first screening tool for Indian students, fresh graduates, and job seekers. Paste a job offer, internship advertisement, email, or recruiter message to inspect classifier output, explicit warning patterns, local guidance passages, and a practical verification checklist.

> JobShield identifies warning signs. It does not verify a sender, employer, domain, or vacancy, and a lower-risk result does not prove legitimacy.

## Features

- TF-IDF + Logistic Regression classification baseline, persisted locally with reproducible held-out split.
- Synthetic fallback examples that are clearly labelled demonstration-only; no invented claim of real-world accuracy.
- Explainable rule-based checks for payment, credentials, identity data, unrealistic promises, pressure, unusual channels, and links.
- Local retrieval with cached sentence-transformer embeddings when available and a TF-IDF cosine-search fallback; the core installation does not install PyTorch/Transformers.
- PDF, TXT, and Markdown ingestion, overlapping chunks, local saved index, and rebuild action.
- Evidence-grounded extractive local explanation by default: relevant passages affect the summary, are cited from stored metadata, and are connected to rule findings only where passage terms support the link. An optional OpenAI-compatible endpoint is disabled unless all settings are explicitly configured.
- Evaluation dashboard, scripts, 30 illustrative assessment records, and pytest suite.

## Architecture

`app.py` contains the Streamlit interface. `src/classifier.py` handles training, persistence, and inference; `indicators.py` keeps explicit rules distinct from the model; `document_ingestion.py`, `embeddings.py`, and `retriever.py` implement local RAG; `rag_pipeline.py` creates a transparent fallback explanation; `verification.py` builds next steps; and `evaluation.py` reports held-out scores.

### Dataset source, labels, and limitations

The project does not bundle or fetch EMSCAD. The source reference is the [Employment Scam Aegean Dataset listing on Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction). Obtain it yourself and check the current license and terms before reuse. Copy an appropriately licensed CSV to `data/emscad.csv`. The loader uses `description` (also accepts `text` or `job_description`) and `fraudulent` (also accepts `is_fraud` or `label`). It drops blank text, invalid labels and duplicate text; maps label 1 to “Potentially suspicious” and 0 to “Lower-risk indicators detected”; and uses a stratified 75/25 split with seed 42. The latter category does not mean legitimate.

Without that CSV, the project uses 80 hand-authored synthetic illustrative messages (40 per class). This is a development fallback, not a public dataset and not representative of real scam prevalence. Its metrics are demonstration-only. Even with real data, source-specific labels, language, concept drift, class imbalance, and domain shift limit generalization. The evaluation set is held out from model fitting; do not use it to tune a model and present it as an unbiased final evaluation.

### Starter knowledge base

Starter entries are short general safety summaries with titles, publisher/source, official URL where available, and unknown dates left blank. They are not presented as quotations. The app links to the [National Cyber Crime Reporting Portal](https://cybercrime.gov.in/), [I4C](https://i4c.mha.gov.in/), and [CERT-In](https://www.cert-in.org.in/). Retrieval similarity is only a ranking signal, not a probability that a passage is true. User-added documents stay in `data/knowledge_base/` and are git-ignored.

## Installation (Windows CMD)

Use Python 3.11. In CMD, from the project folder:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\train_model.py
streamlit run app.py
```

The embedding model is optional at runtime. The app attempts to use `sentence-transformers/all-MiniLM-L6-v2` only if the optional package and weights are already available locally. Otherwise, it uses local TF-IDF vectors without downloading a model.

The core `requirements.txt` pins scikit-learn to **1.5.2** so the persisted classifier artifact and runtime use the same version. Model and index artifacts have version/checksum manifests; missing, stale, or incompatible artifacts are rebuilt from the available local dataset/documents. Do not copy generated `.joblib` files between different scikit-learn versions.

### Optional sentence-transformer backend

The normal core install does not install Sentence Transformers, Transformers, or PyTorch. To add the optional dependency:

```bat
python -m pip install -r requirements-embeddings.txt
```

The default code only loads a model if it is already cached locally; it does not download model weights. To deliberately cache the configured model while online, run:

```bat
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

If the package or model weights are unavailable, the app reports a concise diagnostic and uses the TF-IDF retrieval backend.

## Training and evaluation

```bat
python scripts\train_model.py
python scripts\evaluate_model.py
python scripts\evaluate_model.py --data data\emscad.csv
pytest -q
```

`train_model.py` saves `models/classifier.joblib` plus a version/checksum manifest. Evaluation uses a deterministic stratified hold-out and reports class distribution, split sizes, precision, recall, F1, and confusion matrix in `data/evaluation_results.json`. The script also runs 30 illustrative messages through the actual classifier and records expected/actual categories and correctness in `data/demo_assessments.json`. These scenario checks are a demonstration record, not independent benchmark metrics. The evaluation command retrains and rewrites the model using its indicated dataset. Current evaluation output records the scikit-learn version used.

## Optional LLM configuration

Core classification, retrieval, and template explanation need no API key. To use an OpenAI-compatible chat completions endpoint, copy `.env.example` to `.env` and set all three values:

```text
JOBSHIELD_LLM_API_KEY=your_key
JOBSHIELD_LLM_ENDPOINT=https://provider.example/v1/chat/completions
JOBSHIELD_LLM_MODEL=model-name
```

The configured endpoint receives the recruitment message, classifier result, indicators, and retrieved passages. Review its privacy and retention terms first. API credentials are not committed. Any request failure falls back to the local template.

## Knowledge base usage

Open **Knowledge base**, upload PDF/TXT/Markdown up to 10 MB, and choose **Index uploaded document**. PDF text extraction uses PyMuPDF; scanned image-only PDFs may have no extractable text. Uploaded files and their local index are not committed. **Rebuild local index** reprocesses starter content and documents currently saved in the knowledge-base folder; it removes index references to documents no longer present. Use **Delete selected document** to remove a saved user upload and rebuild the index. The index manifest checks the current source-file inventory and rebuilds when documents or starter guidance change.

## Troubleshooting

- **`python` not found:** install Python 3.11 and enable the Windows PATH option, or use the Python launcher `py -3.11` in place of `python`.
- **Streamlit not found:** activate `.venv` and rerun `python -m pip install -r requirements.txt`.
- **PDF parse or empty text:** ensure the PDF is readable and contains selectable text; image-only scans need OCR, which is not included.
- **Sentence model unavailable:** expected when the optional package or model weights are not present; local TF-IDF retrieval is used, with a diagnostic in the Knowledge base page.
- **Dataset schema error:** confirm the CSV has both supported text and label columns and contains both labels.
- **Stale/incompatible model artifact:** `load_model()` checks its manifest and regenerates a missing, incompatible, corrupt, or checksum-mismatched artifact. To explicitly retrain, run `python scripts\train_model.py`; it uses `data/emscad.csv` if present and otherwise the synthetic demonstration fallback.

## Privacy, limitations, and safe use

Messages are processed locally by default and are not logged or retained by the app; the latest analysis stays in Streamlit session state until the session ends. Uploaded knowledge documents are intentionally stored locally until deleted. Deletion removes the source document and rebuilds the index from remaining files, starters, and their current contents. If an optional external LLM is enabled, the full message and findings are sent to its configured endpoint. No real-time domain checks, company contact, government-database lookup, or identity verification occurs. False positives and false negatives are possible; do not rely on scores for legal, financial, or employment certainty. For suspected cybercrime in India, see https://cybercrime.gov.in/.



- `[Analyze dashboard screenshot placeholder]`
- `[Knowledge base screenshot placeholder]`
- `[Evaluation dashboard screenshot placeholder]`
