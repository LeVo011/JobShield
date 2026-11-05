import requests
from src.classifier import predict
from src.rag_pipeline import explain

def _prediction():
    return {"category":"Potentially suspicious","explanation":"Uncalibrated classifier output."}

def test_local_explanation_uses_passage_text_and_real_metadata(monkeypatch):
    monkeypatch.setattr("src.rag_pipeline.LLM_API_KEY","")
    result=explain(
        "Pay a registration fee and send OTP to secure the offer.",_prediction(),
        [{"name":"Payment request","severity":"high","span":"registration fee"},{"name":"Credential or OTP request","severity":"critical","span":"OTP"}],
        [{"title":"Recruitment Safety Guide","source":"Example Safety Office","date":"2025-02","url":"https://safety.example/guidance","excerpt":"Never pay a recruitment fee to secure a job. Never disclose an OTP or account password to a recruiter.","similarity":0.42}],
    )
    text=result["text"]
    assert "Never pay a recruitment fee" in text
    assert "Recruitment Safety Guide" in text and "Example Safety Office" in text
    assert "https://safety.example/guidance" in text and "2025-02" in text
    assert "Payment request" in text and "Credential or OTP request" in text
    assert "rule-based" in text and "Classifier prediction" in text

def test_different_passage_content_changes_local_explanation(monkeypatch):
    monkeypatch.setattr("src.rag_pipeline.LLM_API_KEY","")
    first={"title":"Fee Advice","source":"Publisher A","excerpt":"Never pay a recruitment fee before confirming the vacancy.","similarity":0.3}
    second={"title":"Credential Advice","source":"Publisher B","excerpt":"Never share an OTP or password with a message sender.","similarity":0.3}
    a=explain("Pay the registration fee",_prediction(),[],[first])["text"]
    b=explain("Share your OTP",_prediction(),[],[second])["text"]
    assert "recruitment fee" in a and "recruitment fee" not in b
    assert "OTP or password" in b and "OTP or password" not in a

def test_irrelevant_passages_are_explicitly_rejected(monkeypatch):
    monkeypatch.setattr("src.rag_pipeline.LLM_API_KEY","")
    result=explain("Pay a fee for a job",_prediction(),[],[{"title":"Weather","source":"Weather Office","excerpt":"Tomorrow will be sunny.","similarity":0.01}])
    assert "Sufficient relevant supporting evidence was not found" in result["text"]
    assert "[1] Weather" not in result["text"]

def test_configured_llm_failure_falls_back_without_exception_details(monkeypatch):
    monkeypatch.setattr("src.rag_pipeline.LLM_API_KEY","test-secret")
    monkeypatch.setattr("src.rag_pipeline.LLM_ENDPOINT","https://provider.invalid/chat")
    monkeypatch.setattr("src.rag_pipeline.LLM_MODEL","example")
    def fail(*args,**kwargs): raise requests.Timeout("secret-token must not appear in UI")
    monkeypatch.setattr(requests,"post",fail)
    result=explain("A recruiter messaged me.",_prediction(),[],[])
    assert result["mode"].startswith("Local template explanation")
    assert "Timeout" in result["diagnostic"]
    assert "secret-token" not in result["text"]

def test_short_input_uncertain():
    assert predict("job?")["category"]=="Uncertain assessment"
