# Data directory

Put an appropriately licensed EMSCAD-compatible CSV at `data/emscad.csv` to train on supplied data. Required fields: `description` (or `text`/`job_description`) and `fraudulent` (or `is_fraud`/`label`), with 0=not labelled fraudulent and 1=labelled fraudulent. No dataset is bundled or downloaded. Source reference: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction. Check current terms/license before use. Otherwise, hand-authored synthetic examples are used for demonstration only.

Knowledge-base starter guidance and index live in `knowledge_base/`. Model/evaluation outputs and uploaded private documents are generated locally.
