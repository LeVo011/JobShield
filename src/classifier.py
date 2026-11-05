"""TF-IDF + logistic regression classification baseline."""
from pathlib import Path
import hashlib
import json
import logging
import os
import shutil
import tempfile
import warnings
import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.exceptions import InconsistentVersionWarning
from src.config import MODEL_PATH,SEED
from src.data_loader import load_dataset
from src.preprocessing import normalize_text

LOGGER=logging.getLogger(__name__)
ARTIFACT_SCHEMA=1

def _manifest_path(model_path):
    return Path(model_path).with_suffix(".manifest.json")

def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def _save_bundle(bundle,model_path):
    model_path=Path(model_path); model_path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp_name=tempfile.mkstemp(prefix=model_path.stem+"_",suffix=".tmp",dir=model_path.parent); os.close(fd)
    tmp=Path(tmp_name); manifest_tmp_path=None
    try:
        joblib.dump(bundle,tmp,compress=3)
        shutil.copyfile(tmp,model_path)
        manifest={"schema":ARTIFACT_SCHEMA,"scikit_learn":sklearn.__version__,"sha256":_sha256(model_path)}
        manifest_path=_manifest_path(model_path)
        fd,manifest_tmp=tempfile.mkstemp(prefix=manifest_path.stem+"_",suffix=".tmp",dir=manifest_path.parent); os.close(fd)
        manifest_tmp_path=Path(manifest_tmp)
        manifest_tmp_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        shutil.copyfile(manifest_tmp_path,manifest_path)
    finally:
        if tmp.exists(): tmp.unlink()
        if manifest_tmp_path and manifest_tmp_path.exists(): manifest_tmp_path.unlink()

def train_model(data_path=None,model_path:Path=MODEL_PATH):
    df,info=load_dataset(data_path); texts=[normalize_text(t) for t in df.text]; y=df.fraudulent.to_numpy()
    Xtr,Xte,ytr,yte=train_test_split(texts,y,test_size=.25,random_state=SEED,stratify=y)
    model=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=30000,sublinear_tf=True)),("clf",LogisticRegression(max_iter=1000,class_weight="balanced",random_state=SEED))]); model.fit(Xtr,ytr)
    probs=model.predict_proba(Xte)[:,list(model.classes_).index(1)]; pred=(probs>=.5).astype(int)
    report=classification_report(yte,pred,labels=[0,1],target_names=["lower_risk_indicators","potentially_suspicious"],output_dict=True,zero_division=0)
    result={"dataset":info,"dataset_size":len(df),"class_counts":{"lower_risk_indicators":int((y==0).sum()),"potentially_suspicious":int((y==1).sum())},"train_size":len(ytr),"test_size":len(yte),"seed":SEED,"scikit_learn_version":sklearn.__version__,"metrics":report,"confusion_matrix":confusion_matrix(yte,pred,labels=[0,1]).tolist(),"warning":"Synthetic demonstration data; metrics are not real-world performance." if info["kind"]=="synthetic_demo" else "Held-out metrics; deployment performance may differ."}
    _save_bundle({"model":model,"metadata":result},Path(model_path)); return result

def _artifact_compatibility_error(model_path):
    model_path=Path(model_path); manifest_path=_manifest_path(model_path)
    if not model_path.is_file(): return "artifact missing"
    if not manifest_path.is_file(): return "artifact manifest missing"
    try:
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return "artifact manifest unreadable"
    if manifest.get("schema")!=ARTIFACT_SCHEMA: return "artifact schema mismatch"
    if manifest.get("scikit_learn")!=sklearn.__version__: return "scikit-learn version mismatch"
    try:
        if manifest.get("sha256")!=_sha256(model_path): return "artifact checksum mismatch"
    except OSError: return "artifact unreadable"
    return None

def load_model(model_path:Path=MODEL_PATH):
    model_path=Path(model_path)
    reason=_artifact_compatibility_error(model_path)
    if reason:
        if model_path.exists(): LOGGER.warning("Ignoring classifier artifact (%s); retraining with scikit-learn %s.",reason,sklearn.__version__)
        train_model(model_path=model_path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error",InconsistentVersionWarning)
            bundle=joblib.load(model_path)
        model=bundle["model"]
        probabilities=model.predict_proba([normalize_text("Please review the role description and confirm an interview time.")])
        if probabilities.shape!=(1,2) or not np.isfinite(probabilities).all(): raise ValueError("Classifier artifact validation failed.")
        if not bundle.get("metadata",{}).get("scikit_learn_version")==sklearn.__version__: raise ValueError("Classifier artifact metadata version mismatch.")
        return bundle
    except Exception as exc:
        LOGGER.warning("Classifier artifact could not be validated (%s); retraining with scikit-learn %s.",type(exc).__name__,sklearn.__version__)
        train_model(model_path=model_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error",InconsistentVersionWarning)
            bundle=joblib.load(model_path)
        return bundle
def predict(text:str,bundle=None):
    if not isinstance(text,str) or not text.strip(): raise ValueError("Enter a recruitment message to analyze.")
    if len(text)>20000: raise ValueError("Message is too long. The limit is 20,000 characters.")
    if len(text.strip())<40: return {"category":"Uncertain assessment","risk":"Insufficient text","score":None,"explanation":"The message is too short for a useful text classification. Add the full message if safe to do so."}
    bundle=bundle or load_model(); model=bundle["model"]; p=float(model.predict_proba([normalize_text(text)])[0,list(model.classes_).index(1)])
    return {"category":"Potentially suspicious" if p>=.5 else "Lower-risk indicators detected","risk":"Elevated model signal" if p>=.5 else "Lower model signal","score":p,"explanation":"TF-IDF and logistic regression estimate similarity to labelled training examples. This score is not calibrated and is not a probability that the sender is fraudulent."}
