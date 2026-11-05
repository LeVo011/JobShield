import pytest
import json
import sklearn
from src.classifier import train_model,predict,load_model,_manifest_path
from src.config import MODEL_PATH
def test_train_save_load_predict(tmp_path):
    path=tmp_path/"model.joblib"; train_model(model_path=path); from src.classifier import load_model
    result=predict("Pay a registration fee today to get guaranteed job selection.",load_model(path))
    assert result["category"] in {"Potentially suspicious","Lower-risk indicators detected"}; assert 0<=result["score"]<=1
def test_empty_short_and_long():
    with pytest.raises(ValueError): predict("  ")
    assert predict("job?")["category"]=="Uncertain assessment"
    with pytest.raises(ValueError): predict("x"*20001)

def test_actual_persisted_artifact_loads_and_predicts():
    bundle=load_model()
    assert bundle["metadata"]["scikit_learn_version"]==sklearn.__version__
    assert predict("Please send a registration fee to secure the internship.",bundle)["score"] is not None

def test_incompatible_artifact_manifest_is_regenerated(tmp_path):
    path=tmp_path/"legacy.joblib"
    result=train_model(model_path=path)
    manifest_path=_manifest_path(path)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scikit_learn"]="0.0-incompatible"
    manifest_path.write_text(json.dumps(manifest),encoding="utf-8")
    bundle=load_model(path)
    assert bundle["metadata"]["scikit_learn_version"]==sklearn.__version__
    assert predict("An ordinary interview invitation is scheduled for next week.",bundle)["score"] is not None

def test_missing_artifact_is_regenerated(tmp_path):
    path=tmp_path/"missing.joblib"
    bundle=load_model(path)
    assert path.exists() and _manifest_path(path).exists()
    assert bundle["metadata"]["scikit_learn_version"]==sklearn.__version__
