# JobShield Technical Audit

**Audit date:** 2026-11-05  
**Scope:** Initial audit findings plus the authorized remediation and re-verification performed after that audit. Source, generated classifier/index artifacts, tests, README, and this audit were updated; no academic report was written. The optional external LLM was not configured.

## Executive summary

JobShield has a Streamlit UI, a scikit-learn text classifier and training/evaluation pipeline, explicit warning rules, local document ingestion/retrieval, tailored checklist generation, and an extractive local explanation. The current index uses TF-IDF with NumPy cosine similarity; no sentence-transformer weights were available. The local explanation now selects content from relevant retrieved passages, cites stored title/source/date/URL metadata, and distinguishes evidence, rule matches, model output, and inference. Optional LLM support remains.

The recorded evaluation uses **80 hand-authored synthetic examples**, not EMSCAD. Its deterministic held-out accuracy is 0.85 (17/20); this is a pipeline demonstration, not evidence of generalization to real recruitment messages. A fresh persisted model was trained under scikit-learn 1.5.2 and passed five inference probes. The updated automated suite passes 24 tests.

**Resolved:** the old classifier artifact that failed under scikit-learn 1.5.2 has been regenerated. Its manifest records scikit-learn 1.5.2 and a SHA-256 checksum. Loading now validates the manifest and a prediction probe; missing, mismatched, corrupt, or incompatible artifacts are retrained instead of being blindly reused.

**Resolved for the current local index:** a rebuild from present source files replaced the stale index. The resulting index contains only four starter chunks; the previous PDF title/content is absent. Index manifests fingerprint current source files and starter content, so startup detects removed or changed documents. The UI now offers deletion of saved uploads and rebuilds after deletion.

## 1. Complete project structure

The following is the project structure observed, excluding the internal contents of `.git/` and transient Python bytecode/cache directories. Generated files marked **ignored** are present locally but excluded by `.gitignore`.

```text
JobShield/
├── .env.example
├── .gitignore
├── README.md
├── PROJECT_AUDIT.md                 # this audit
├── app.py
├── requirements.txt
├── requirements-embeddings.txt      # optional heavy semantic backend
├── data/
│   ├── README.md
│   ├── demo_assessments.json        # generated, ignored
│   ├── evaluation_results.json      # generated, ignored
│   └── knowledge_base/
│       ├── .gitkeep
│       ├── index.joblib             # generated, ignored; current TF-IDF index
│       └── index.manifest.json      # generated, ignored; version/hash/source inventory
├── models/
│   ├── .gitkeep
│   ├── classifier.joblib            # generated, ignored; trained with sklearn 1.5.2
│   └── classifier.manifest.json     # generated, ignored; schema/version/checksum
├── scripts/
│   ├── evaluate_model.py
│   └── train_model.py
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── config.py
│   ├── data_loader.py
│   ├── document_ingestion.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── indicators.py
│   ├── preprocessing.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── ui_state.py
│   └── verification.py
└── tests/
    ├── conftest.py
    ├── test_classifier.py
    ├── test_indicators.py
    ├── test_preprocessing.py
    ├── test_rag_pipeline.py
    ├── test_retrieval.py
    ├── test_ui_state.py
    └── test_verification.py
```

No `.venv/`, `.env`, `data/emscad.csv`, or uploaded source PDF is present at the corrected audit time. The generated classifier and clean index plus their manifests are present locally and ignored by Git. A `.git/` directory exists. Project files remain untracked in this workspace; they have not been committed. `.gitignore` excludes model/index artifacts and manifests, evaluation JSON, `.env`, uploaded knowledge-base documents, virtual environments, and caches.

## 2. Installed technology stack

Versions are from the active Python 3.11.9 environment on this machine, not from a project virtual environment. Core dependencies are mostly ranged; scikit-learn is pinned to 1.5.2. There is no full lockfile.

| Package | Installed version | Role |
|---|---:|---|
| Python | 3.11.9 | Runtime |
| Streamlit | 1.66.1 | UI |
| scikit-learn | 1.5.2 | TF-IDF, Logistic Regression, split and metrics |
| pandas | 2.2.3 | Dataset handling and dashboard tables |
| NumPy | 2.2.6 | Vector matrices and cosine similarity |
| joblib | 1.4.2 | Persisting classifier and retrieval index |
| python-dotenv | 1.2.1 | Optional `.env` configuration |
| PyMuPDF | 1.28.2 | PDF text extraction |
| requests | 2.31.0 | Optional external LLM request |
| pytest | 8.4.2 | Automated tests |
| sentence-transformers | 5.7.0 | Installed in this shared environment as an optional package; no model weights are cached |
| transformers | 5.22.1 | Installed optional-stack dependency; not in core requirements |
| torch | 2.10.0 | Installed optional-stack dependency; not in core requirements |
| huggingface-hub | 2.4.0 | Installed optional-stack dependency; not in core requirements |

No FAISS package is installed or used. Sentence-transformers has been removed from core `requirements.txt` and placed in `requirements-embeddings.txt`; a normal core installation therefore avoids installing PyTorch/Transformers. Current shared-environment versions above reflect packages installed during earlier setup, not the corrected core dependency set. The optional model is only loaded if its weights are already cached; the app does not download weights automatically.

`python -m pip check` **failed** in the shared global environment. Reported conflicts include Gradio/gradio-client requiring `huggingface-hub<2.0` while 2.4.0 is installed; TensorFlow requiring NumPy `<2.0` and protobuf `<5` while NumPy 2.2.6 and protobuf 6.33.5 are installed; plus unrelated EasyOCR/OpenCV, spaCy/Weasel/Typer, and MLflow/cryptography conflicts. These are not direct core requirements; a clean project virtual environment is required for a meaningful dependency check. `requirements.txt` now pins scikit-learn to 1.5.2.

## 3. NLP model architecture and training procedure

Source: `src/classifier.py`, `src/preprocessing.py`, `src/data_loader.py`, `scripts/train_model.py`.

1. Text is normalized using Unicode NFKC, NUL removal, URL/email replacement with tokens, whitespace folding, and lowercasing.
2. A scikit-learn `Pipeline` contains `TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000, sublinear_tf=True)` followed by `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`.
3. `train_test_split` uses `test_size=0.25`, `random_state=42`, and `stratify=y`. The vectorizer is inside the pipeline, so its vocabulary is fit on training text only.
4. The model predicts the positive-class score and applies a fixed 0.5 cutoff. Inputs shorter than 40 characters return `Uncertain assessment`; empty and over-20,000-character inputs raise validation errors.
5. The serialized bundle stores the pipeline, evaluation metadata, and scikit-learn version. A separate JSON manifest records schema, version, and artifact SHA-256. `load_model()` checks the manifest before unpickling, validates a probe prediction, and retrains missing, incompatible, corrupt, or checksum-mismatched artifacts. Manifest checks protect against accidental compatibility errors; joblib files must still be trusted local artifacts.

The displayed score is explicitly uncalibrated by the UI. It is a model score from `predict_proba`, not a calibrated probability that a message is fraudulent. “Lower-risk indicators detected” means only that this model score fell below the cutoff.

### Dataset and provenance

**Actual dataset used:** the 80 hand-authored examples in `src/data_loader.py`, described in code as synthetic development examples. This is not a downloaded or public dataset and has no external original source or separately declared license in the project. The EMSCAD Kaggle URL in `src/data_loader.py`, `README.md`, and `data/README.md` is a reference for a possible user-supplied dataset; it is not the provenance of the actual model data. EMSCAD's current license/terms were not verified in this audit.

| Actual class | Label | Records |
|---|---:|---:|
| Lower-risk indicators detected | 0 | 40 |
| Potentially suspicious | 1 | 40 |
| **Total** |  | **80** |

The synthetic examples are manually authored recruitment-like messages, using fictional examples in the UI. They are not measured real-world cases and should not be used to estimate scam prevalence or deployment performance.

If `data/emscad.csv` or a path passed to the loader exists, the loader accepts `description`/`text`/`job_description` and `fraudulent`/`is_fraud`/`label`. It coerces labels numerically, drops invalid labels, casts to integer, **clips values to 0 or 1**, drops blank text, removes duplicate text, and requires at least 10 rows with both classes. It does not verify the file's license or provenance. Clipping means unexpected numeric labels can be silently coerced into one of the two classes instead of rejected. EMSCAD label mapping is assumed to be 1=suspicious and 0=lower-risk; no other columns are used.

No actual EMSCAD file was present. No external dataset records, class balance, or license can therefore be reported for EMSCAD.

## 4. Actual split and held-out metrics

Verified by a fresh `train_model()` invocation writing only to a temporary directory and by comparison with `data/evaluation_results.json`:

- Dataset: 80 synthetic examples, 40 per class.
- Training set: 60 rows.
- Held-out test set: 20 rows, 10 per class.
- Split: stratified 75/25, seed 42.
- Accuracy: **0.85** (17 correct of 20).

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Lower-risk indicators detected | 0.8182 | 0.9000 | 0.8571 | 10 |
| Potentially suspicious | 0.8889 | 0.8000 | 0.8421 | 10 |
| Macro average | 0.8535 | 0.8500 | 0.8496 | 20 |
| Weighted average | 0.8535 | 0.8500 | 0.8496 | 20 |

Confusion matrix, rows=true `[lower-risk, suspicious]`, columns=predicted `[lower-risk, suspicious]`:

```text
[[9, 1],
 [2, 8]]
```

These are actual metrics for the generated synthetic holdout only. The project implements no simple baseline comparison. No calibration curve, confidence interval, cross-validation, external benchmark, or real-data evaluation is implemented.

`src/evaluation.py` additionally checks 30 separate hand-authored scenarios against hand-written expected labels. The actual check was **29/30**; one message expected lower-risk was classified suspicious at score 0.5043: “The recruiter invited me to discuss the advertised software internship by phone.” The scenarios are not independent external validation. Recorded inputs/outputs are in ignored `data/demo_assessments.json`.

## 5. Embedding model and vector index

Sources: `src/embeddings.py`, `src/retriever.py`.

- Preferred configured model name: `sentence-transformers/all-MiniLM-L6-v2`.
- `SentenceTransformer` is instantiated with `local_files_only=True`; no model download is attempted. Missing optional packages and unavailable/corrupt local weights now produce sanitized diagnostics before switching to TF-IDF.
- The **actual persisted index backend is TF-IDF**, verified from `data/knowledge_base/index.joblib`. After rebuilding, it contains a `(4, 244)` float32 matrix: four starter chunks and 244 vector features.
- Search transforms the query with the stored TF-IDF vectorizer and ranks rows with NumPy cosine similarity. Only positive-score hits are returned, up to the requested `k` (four in the UI).
- Index rows, excerpts, metadata, vectorizer, and matrix are serialized in one local `joblib` artifact. A JSON manifest checks schema, scikit-learn version, checksum, starter-data digest, and current uploaded-document inventory; stale or incompatible indexes rebuild from present source files. This is local vector search, not a dedicated vector database. FAISS is not present.

The previous persisted index emitted scikit-learn 1.9.1-to-1.5.2 warnings and contained removed-document chunks. It was rebuilt under 1.5.2; the current manifest matches the runtime, and the stale PDF row/content is absent. The Knowledge base page now reports the active backend and fallback diagnostic.

## 6. Document ingestion, chunking, and retrieval

Source: `src/document_ingestion.py`, `src/retriever.py`, `app.py`.

- Accepted suffixes: `.pdf`, `.txt`, `.md`, `.markdown`.
- Maximum upload size: 10 MiB. Text/Markdown are decoded as UTF-8 with BOM support. PDFs are parsed from bytes using PyMuPDF. Image-only PDF OCR is not supported.
- Text cleanup removes NULs, folds horizontal spaces, and reduces runs of three or more newlines.
- Chunking is by **words**, not model tokens or characters: 800 words per chunk with 120-word overlap (680-word stride).
- User uploads are written to `data/knowledge_base/<basename>` and the index is rebuilt. Re-uploading the same basename overwrites the existing file. The Knowledge base UI now offers deletion of uploaded files and rebuilds after deletion.
- User-document metadata is title from filename stem, source “User-provided document,” unknown date, no URL, filename, excerpt, and chunk index. Starter metadata includes title/source/date/URL; dates are null where unknown.
- `rebuild()` reconstructs the full index from starter text plus parseable supported files currently in the knowledge-base directory; it does not merge old rows. A manifest compares current source filenames/content hashes and starter content at startup, triggering rebuild if files disappear or change. `search()` returns excerpt, source metadata, chunk, and similarity ranking.

The pre-fix index contained two chunks from `JD_GET FY 26-27 (B) (1).pdf` after its source PDF had disappeared. A corrected rebuild removed those chunks; the current index contains only the four starter passages. The new index manifest will trigger a rebuild when a source document is removed or changed.

## 7. Is RAG implemented and working?

**Retrieval is genuinely implemented and working.** In an audit query against the persisted index, local cosine retrieval returned ranked starter/user-document excerpts and similarities. Ingestion, chunking, persistence, query transformation, and retrieval are executable local code paths.

**The local retrieval-augmented explanation is now implemented and verified in the default configuration.** `src/rag_pipeline.py` filters low-scoring/non-overlapping passages, selects relevant sentences from actual excerpt content, includes citations assembled from passage title/source/date/URL metadata, and explicitly relates those passages to rule findings only when terms overlap. The explanation separates model prediction, rule findings, retrieved content, and inference. When no passage meets relevance checks, it states that relevant supporting evidence was not found and cites nothing. Tests prove that changing passage text changes the output, source metadata is used, and irrelevant low-score material is rejected. This is an extractive/template explanation, not generative summarization. The optional LLM path passes the same relevant excerpts but remains unconfigured/unverified.

The four starter entries are hand-authored summaries with source metadata, not copied official quotations. The project does not automatically fetch or refresh those pages. Similarity is correctly labelled as ranking rather than factual probability, but it cannot establish source authority or claim correctness.

## 8. LLM integration versus template fallback

The normal path is **`Local template explanation`**, not an LLM. `src/rag_pipeline.py` enables an optional OpenAI-compatible chat-completions request only if all of `JOBSHIELD_LLM_API_KEY`, `JOBSHIELD_LLM_ENDPOINT`, and `JOBSHIELD_LLM_MODEL` are non-empty. No `.env` file or credentials were present, and no LLM request was made or tested.

When configured, the adapter sends the full recruitment message, classifier result, detected indicators, and relevant retrieved excerpts to the configured endpoint with a 20-second timeout. Request failures fall back to the local template and expose only the exception class (not exception text, endpoint, token, or response details). The UI labels the mode. The application does not redact message content before the configured request; external data handling is the operator's responsibility.

## 9. Complete system workflow

1. The Analyze page accepts text in a Streamlit text area (maximum 20,000 characters), or fills one of six illustrative examples.
2. The classifier validates emptiness/length, returns uncertainty below 40 characters, normalizes the text, vectorizes it, and returns a class and uncalibrated score.
3. The separate indicator rules scan the raw input and return at most one match per configured rule category, with span, severity, reason, and next step.
4. The retrieval query combines the original message with indicator names; the local index returns up to four positive-similarity chunks.
5. The local template selects source sentences from relevant retrieved passages and cites their actual metadata, or explicitly reports insufficient relevant evidence. If configured and successful, the optional LLM receives those excerpts; request failure returns the local explanation with a sanitized diagnostic.
6. The results UI shows classifier category/score, model caveat, observed rule matches, explanation mode/text, relevant retrieved excerpts and metadata, and uncertainty warning. If the user edits the input, old results are hidden until the new input is analyzed.
7. Checklist generation adds general independent-verification steps, conditionally inserts steps for payment/credential/identity findings, and includes `https://cybercrime.gov.in/`.
8. Checklist widget keys include a per-analysis generation so checkmarks from prior messages do not carry forward. The analysis is held in Streamlit session state for the current session; there is no persistent analysis database or message analytics pipeline.

## 10. Implemented, partial, and absent features

| Status | Feature | Audit finding |
|---|---|---|
| Implemented | Streamlit dashboard pages | Analyze, Knowledge base, Evaluation, About & limitations are present. |
| Implemented | TF-IDF + Logistic Regression pipeline | Training, manifest-backed persistence, compatibility validation/repair, prediction, uncertain short-input handling. Current persisted artifact was freshly trained and checked under sklearn 1.5.2. |
| Implemented | Synthetic development data | 80 authored rows, balanced; explicitly marked demonstration-only. |
| Implemented | Rule-based indicators | Payment, credentials/OTP, identity/banking, high/guaranteed claims, urgency, unusual channel, links. |
| Implemented | Local TXT/PDF/Markdown ingestion | Size and suffix checks; UTF-8 text parsing and PyMuPDF PDF text extraction. |
| Implemented | Local vector retrieval | Current backend is TF-IDF with NumPy cosine ranking; persisted joblib state and source/version/checksum manifest. |
| Implemented | Verification checklist | Generic checks plus indicator-specific additions and official reporting link. |
| Implemented | Local template explanation | Extracts relevant sentence(s) from retrieved text, cites stored metadata, distinguishes findings/prediction/evidence/inference, and reports insufficient evidence. |
| Implemented | Local RAG retrieval-to-evidence path | Actual passages affect the local explanation; tests cover content sensitivity, metadata citations, and irrelevant evidence rejection. It is extractive rather than abstractive generation. |
| Partial | Sentence-transformer embeddings | Optional package path exists, but weights are not cached and current index uses TF-IDF; core install no longer installs the package. |
| Partial | Evaluation | Real held-out split metrics are calculated, but only on synthetic data; no baseline or external validation. |
| Implemented | User document management | Add, delete, explicit rebuild, and startup detection of changed/removed sources; stale rows are replaced from current sources. |
| Partial | PDF robustness | Text PDFs supported; OCR/scanned documents unsupported. |
| Partial | LLM adapter | OpenAI-compatible request path exists; no configuration/live request/provider compatibility test was performed. Failure fallback is tested and sanitized. |
| Not implemented | EMSCAD download/use | No real dataset file was present or used; Kaggle URL is documentation only. |
| Not implemented | Live employer/domain verification | No website, email-domain, government database, or company contact checks occur. |
| Not implemented | Baseline model comparison | Evaluation UI says none is shown. |
| Not implemented | Source-document refresh/verification | Starter source summaries are static; dates are unknown; pages are not fetched. |
| Not implemented | OCR, multilingual support, calibrated risk, or demographic analysis | No OCR, language detection/multilingual model, score calibration, or demographic inference exists. |

## 11. Automated tests and execution results

Audit command: `python -B -m pytest -q -p no:cacheprovider`  
Corrected result: **24 passed in 15.27 seconds; 0 failures.** Earlier intermediate runs exposed Windows rename restrictions, two RAG token-set defects, and a diagnostic-preservation assertion; those were fixed before this final result.

Tests cover normalization, non-text input, temporary and actual persisted model loading/prediction, missing and incompatible artifact regeneration, empty/short/long inputs, payment and OTP indicators, ordinary copy, TXT parsing/chunking, retrieval, unsupported extension, size limit, empty upload, stale-index cleanup after rebuild and on startup, document deletion and persisted-row removal, path traversal rejection, local RAG text sensitivity, metadata citations, irrelevant passage rejection, sanitized LLM-failure fallback, uncertain short input, checklist/report link generation, per-analysis checkbox keys, and hiding stale results when input changes.

Coverage gaps: no PDF parse success test, no live configured LLM test, no full browser-driven Streamlit interaction test, and no real-data evaluation test. Streamlit `AppTest` hung before returning a test page in this environment; desktop browser automation was denied access to Chrome. The server health and root HTTP routes were checked separately.

The persisted `models/classifier.joblib` was regenerated with scikit-learn **1.5.2**. The updated suite loads the actual configured `MODEL_PATH`; five sample messages were also run against the persisted artifact: fee request → suspicious (0.6653), OTP/password request → suspicious (0.6140), ordinary interview scheduling → lower-risk (0.4157), unverified Aadhaar/bank request → suspicious (0.6438), and short ambiguous message → uncertain (no score). These are demonstration probes, not evaluation metrics.

`python -m pip check` returned exit code 1 because of shared-environment dependency conflicts listed in the technology section. This is separate from the pytest result.

## 12. Application startup and runtime requirements

Documented Windows CMD startup from the project root:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts\train_model.py
streamlit run app.py
```

The corrected server was launched as `python -m streamlit run app.py --server.headless true --server.port 8501 --server.address 127.0.0.1`; `/_stcore/health` returned `200 ok` and `/` returned HTTP 200 with `text/html`. Browser-driven page navigation/core interactions were not completed: Streamlit `AppTest` stalled in this environment and desktop automation was denied access to Chrome. The source branches and underlying interactions are covered by tests, but this is not a substitute for a full browser demo check.

No GPU or paid API is required for the implemented classifier or TF-IDF retrieval. CPU is sufficient for the current core path. Optional sentence embeddings, if installed and cached, can be CPU-hosted but add PyTorch/Transformers and may increase startup and memory cost. The project has no measured minimum RAM/disk requirement. The current TF-IDF matrix is small; it is densified to NumPy arrays and may use substantial memory for a large knowledge base.

## 13. Privacy and security implementation

**Implemented safeguards:**

- Messages are not written to an application history or explicit log by source code; analysis lives in Streamlit session state.
- No external service is called unless all optional LLM environment variables are configured.
- `.env`, model/index artifacts, uploaded documents, generated JSON, and caches are ignored by Git.
- File suffix and 10 MiB size are checked before parsing; upload names are reduced to a basename before storage.
- UI and README disclose external LLM transmission and warn that no real verification occurs.
- API keys are configured through environment variables; none were found in the inspected project files.

**Residual risks:**

- Configured LLM requests send the entire raw message and findings without redaction.
- User uploads are intentionally retained on disk until deleted with the Knowledge base control. Deletion removes the file and rebuilds the index; manifest validation also detects files removed or changed outside the app.
- The stale PDF chunks found during the original audit have been removed from the current local index. Rebuild/deletion tests verify that absent-document excerpts do not remain in the rewritten artifact.
- `joblib.load` loads pickle-based artifacts and must only be used on trusted local model/index files; loading a malicious pickle can execute code.
- PDF parsing handles size and parse errors but does not implement OCR or a dedicated resource-exhaustion defense for unusually complex PDFs.
- Streamlit session state can retain the last message for the lifetime of a browser session.

## 14. Known limitations and technical issues

### Remaining demonstration/evaluation risks

1. **Real dataset not supplied.** The model and metrics remain based on synthetic examples. If the rubric requires a public fraud dataset, provide and license-check an EMSCAD-compatible CSV and retrain/evaluate it before making performance claims.
2. **Full browser interaction pass remains unverified.** The server health and root route work, but desktop browser automation was denied by policy and Streamlit `AppTest` stalled before returning. Manually click every page before the academic demo.
3. **Use an isolated virtual environment.** `pip check` still fails in the shared global environment due unrelated installed-package conflicts; core `requirements.txt` is now lighter and pins scikit-learn 1.5.2.

### Other review/evaluation risks

- No real dataset was used; the displayed 85% accuracy is only on 20 synthetic holdout rows.
- A synthetic scenario check had one false positive (29/30 expected labels matched).
- Model estimates are not calibrated. The 0.5 cutoff is fixed, and the UI score could still be mistaken for a risk probability despite the caveat.
- Default local explanation is extractive and heuristic, not abstractive/generative. Relevance uses a local similarity floor and lexical overlap; it can still omit semantically related passages or include tangential context.
- Sentence Transformers model weights are not cached, so the current RAG backend remains TF-IDF. The optional LLM path has not been verified against a real provider.
- Checklist keys now include a per-analysis generation, and results are hidden when the input changes until re-analysis.
- CSV label clipping can silently coerce unexpected labels; label schema/license are not independently validated.
- Uploaded-document title is filename-derived and publication/source metadata is not parsed or verified.
- The evaluation command retrains and overwrites the project model/evaluation outputs; it should not be casually run against a valuable artifact without a backup.
- Project files were untracked in Git at audit time, which can affect handoff or archival if a commit/package is expected.

## 15. Exact source-file map

| Module | Source file(s) | Responsibility |
|---|---|---|
| UI and session state | `app.py` | Streamlit pages, inputs, results, upload controls, evaluation display, session analysis. |
| Paths and environment | `src/config.py`, `.env.example` | Local paths, dotenv, model/index settings, optional LLM settings, limits. |
| Dataset | `src/data_loader.py`, `data/README.md` | Synthetic examples and supported optional labelled CSV loading. |
| Text normalization | `src/preprocessing.py` | Unicode/whitespace normalization and URL/email token replacement. |
| Classification | `src/classifier.py`, `scripts/train_model.py` | TF-IDF/LR construction, splitting, fitting, persistence, loading, prediction. |
| Warning indicators | `src/indicators.py` | Explicit regex checks, spans, severities, explanations, recommended actions. |
| Embeddings | `src/embeddings.py` | Cached local sentence-transformer attempt and TF-IDF fallback. |
| Document parsing/chunks | `src/document_ingestion.py` | PDF/TXT/Markdown extraction, validation, word chunks and overlap. |
| Local index/retrieval | `src/retriever.py` | Starter sources, user docs, vectorization, cosine ranking, joblib persistence. |
| Explanation/optional LLM | `src/rag_pipeline.py` | Local evidence-aware template and optional OpenAI-compatible adapter. |
| Verification checklist | `src/verification.py` | General and indicator-specific safety steps. |
| Evaluation | `src/evaluation.py`, `scripts/evaluate_model.py` | Held-out metrics and 30 illustrative scenario checks. |
| Streamlit state helpers | `src/ui_state.py` | Per-analysis widget keys and current-input/result matching. |
| Tests | `tests/conftest.py`, `tests/test_*.py` | Import setup and unit/integration tests described above. |
| Setup and architecture docs | `README.md`, `requirements.txt`, `requirements-embeddings.txt`, `.gitignore` | Core and optional installs, usage, package versions, ignored local artifacts. |

## Audit verification commands

Commands used during this audit included:

```text
python -B -m pytest -q -p no:cacheprovider
python -B scripts/train_model.py
python -B -c "from src.classifier import load_model, predict; ... five sample messages ..."
python -m pip check
python -m streamlit run app.py --server.headless true --server.port 8501 --server.address 127.0.0.1
GET http://127.0.0.1:8501/_stcore/health
GET http://127.0.0.1:8501/
```

The final tests, `scripts/train_model.py`, `scripts/evaluate_model.py`, persisted-artifact inference, RAG retrieval/explanation, default-index cleanup, and HTTP health/root checks were run. The five-message probes were fee, OTP/password, ordinary interview, identity/banking, and short ambiguous text. The RAG query returned three vector-ranked candidates; lexical relevance filtering excluded the irrelevant CERT-In passage and used the two passages whose text overlapped the payment subject. The 30-case evaluation record is 29/30 from the deterministic scenario run. Full browser navigation remains unverified because UI automation was denied and AppTest stalled.
