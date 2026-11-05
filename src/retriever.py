"""Persisted local cosine-similarity retrieval with source metadata."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import warnings
import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
from src.config import KB_DIR
from src.embeddings import Embedder
from src.document_ingestion import extract_text,chunk_text
STARTERS=[
{"title":"National Cyber Crime Reporting Portal","source":"Government of India","date":None,"url":"https://cybercrime.gov.in/","text":"The National Cyber Crime Reporting Portal is the Government of India's official portal for reporting cybercrime. Preserve relevant messages, payment records, and other evidence when making a report. For emergencies, use local emergency services."},
{"title":"I4C: Cyber safety awareness","source":"Indian Cyber Crime Coordination Centre","date":None,"url":"https://i4c.mha.gov.in/","text":"The Indian Cyber Crime Coordination Centre supports coordinated action against cybercrime. Use official channels and verify information independently. Do not share OTPs, passwords, or financial credentials with callers or message senders."},
{"title":"CERT-In","source":"Indian Computer Emergency Response Team","date":None,"url":"https://www.cert-in.org.in/","text":"CERT-In is India's national nodal agency for responding to computer security incidents. Follow current advisories published on its official website and report relevant incidents through appropriate official channels."},
{"title":"Recruitment message verification checklist","source":"JobShield project guidance (general safety advice)","date":None,"url":None,"text":"Verify a vacancy on the employer's careers website reached independently. Confirm a recruiter's identity using company contact information independently obtained from the official website. Treat requests for payment, OTPs, passwords, PINs, or card security codes as high concern. Inspect links without entering credentials and preserve messages and transaction details if fraud is suspected. A warning sign is not proof of fraud."}]
INDEX_SCHEMA=1

def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

class LocalRetriever:
    def __init__(self,directory:Path=KB_DIR):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True); self.state_path=self.directory/"index.joblib"; self.manifest_path=self.directory/"index.manifest.json"; self.embedder=None; self.rows=[]; self.matrix=None; self.diagnostic=""; self.document_diagnostics=[]
        reason=self._manifest_error()
        if not reason:
            try:
                import joblib
                with warnings.catch_warnings():
                    warnings.simplefilter("error",InconsistentVersionWarning)
                    state=joblib.load(self.state_path)
                self.embedder=state["embedder"]; self.rows=state["rows"]; self.matrix=state["matrix"]
                if len(self.rows)!=len(self.matrix): raise ValueError("Index row and matrix counts differ.")
            except Exception as exc: reason=f"saved index could not be validated ({type(exc).__name__})"
        if reason:
            self.rebuild(diagnostic=f"Rebuilt the local index because the {reason}.")
        elif getattr(self.embedder,"diagnostic",""):
            self.diagnostic=self.embedder.diagnostic

    def _manifest_error(self):
        if not self.state_path.is_file(): return "index is missing"
        if not self.manifest_path.is_file(): return "index manifest is missing"
        try: manifest=json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError): return "index manifest is unreadable"
        if manifest.get("schema")!=INDEX_SCHEMA: return "index schema changed"
        if manifest.get("scikit_learn")!=sklearn.__version__: return "scikit-learn version changed"
        if manifest.get("sources")!=self._source_fingerprint(): return "knowledge documents changed"
        starter_digest=hashlib.sha256(json.dumps(STARTERS,sort_keys=True,ensure_ascii=False).encode("utf-8")).hexdigest()
        if manifest.get("starter_digest")!=starter_digest: return "starter guidance changed"
        try:
            if manifest.get("sha256")!=_sha256(self.state_path): return "index checksum changed"
        except OSError: return "index artifact is unreadable"
        return None

    def _source_fingerprint(self):
        sources=[]
        for path in sorted(self.directory.iterdir()):
            if path.suffix.lower() not in {".pdf",".txt",".md",".markdown"} or not path.is_file(): continue
            try: sources.append({"name":path.name,"sha256":_sha256(path)})
            except OSError: sources.append({"name":path.name,"sha256":"unreadable"})
        return sources
    def _documents(self):
        rows=list(STARTERS)
        for p in sorted(self.directory.iterdir()):
            if p.suffix.lower() not in {".pdf",".txt",".md",".markdown"}: continue
            try: rows.append({"title":p.stem,"source":"User-provided document","date":None,"url":None,"text":extract_text(p.name,p.read_bytes()),"filename":p.name})
            except Exception as exc: self.document_diagnostics.append(f"A saved knowledge document was skipped ({type(exc).__name__}).")
        return rows
    def rebuild(self,diagnostic=""):
        self.document_diagnostics=[]
        rows=[]
        for doc in self._documents():
            for i,chunk in enumerate(chunk_text(doc["text"])): rows.append({k:v for k,v in doc.items() if k!="text"}|{"excerpt":chunk,"chunk":i})
        self.rows=rows; self.embedder=Embedder(); self.matrix=self.embedder.fit_transform([r["excerpt"] for r in rows]) if rows else np.zeros((0,0),dtype="float32")
        import joblib
        fd,tmp_name=tempfile.mkstemp(prefix="index_",suffix=".tmp",dir=self.directory); os.close(fd); tmp=Path(tmp_name); manifest_tmp_path=None
        try:
            joblib.dump({"embedder":self.embedder,"rows":self.rows,"matrix":self.matrix},tmp,compress=3); shutil.copyfile(tmp,self.state_path)
            manifest={"schema":INDEX_SCHEMA,"scikit_learn":sklearn.__version__,"sha256":_sha256(self.state_path),"sources":self._source_fingerprint(),"starter_digest":hashlib.sha256(json.dumps(STARTERS,sort_keys=True,ensure_ascii=False).encode("utf-8")).hexdigest()}
            fd,manifest_tmp=tempfile.mkstemp(prefix="index_manifest_",suffix=".tmp",dir=self.directory); os.close(fd)
            manifest_tmp_path=Path(manifest_tmp); manifest_tmp_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8"); shutil.copyfile(manifest_tmp_path,self.manifest_path)
        finally:
            if tmp.exists(): tmp.unlink()
            if manifest_tmp_path and manifest_tmp_path.exists(): manifest_tmp_path.unlink()
        self.diagnostic="; ".join(part for part in (diagnostic,self.embedder.diagnostic) if part)
        return rows
    def add_document(self,name,data):
        text=extract_text(name,data)
        if not text: raise ValueError("No extractable text was found in the document.")
        dest=self.directory/Path(name).name; dest.write_bytes(data); self.rebuild(); return len(chunk_text(text))
    def delete_document(self,name):
        safe_name=Path(name).name
        if safe_name!=name or Path(safe_name).suffix.lower() not in {".pdf",".txt",".md",".markdown"}: raise ValueError("Select a saved PDF, TXT, or Markdown document.")
        dest=self.directory/safe_name
        if not dest.is_file(): return False
        dest.unlink(); self.rebuild(); return True
    def search(self,query,k=4):
        if not query.strip() or not self.rows: return []
        q=self.embedder.transform([query])[0]; denom=np.linalg.norm(self.matrix,axis=1)*np.linalg.norm(q)
        scores=np.divide(self.matrix@q,denom,out=np.zeros(len(denom),dtype="float32"),where=denom!=0); inds=np.argsort(scores)[::-1][:k]
        return [{**self.rows[i],"similarity":float(scores[i])} for i in inds if scores[i]>0]
    def documents(self):
        docs={}
        for r in self.rows:
            key=r.get("filename") or r["title"]
            docs.setdefault(key,{"title":r["title"],"filename":r.get("filename"),"chunks":0,"source":r["source"]})["chunks"]+=1
        return list(docs.values())
