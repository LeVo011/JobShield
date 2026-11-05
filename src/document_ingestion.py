"""Parse PDF/TXT/Markdown documents and split them into overlapping chunks."""
from pathlib import Path
import re
SUPPORTED={".pdf",".txt",".md",".markdown"}
def extract_text(name,data,max_bytes=10*1024*1024):
    suffix=Path(name).suffix.lower()
    if suffix not in SUPPORTED: raise ValueError("Supported document types: PDF, TXT, and Markdown.")
    if len(data)>max_bytes: raise ValueError("Document exceeds the 10 MB limit.")
    if suffix==".pdf":
        try:
            import fitz
            doc=fitz.open(stream=data,filetype="pdf"); text="\n".join(p.get_text() for p in doc); doc.close()
        except Exception as e: raise ValueError(f"Could not parse PDF ({type(e).__name__}).") from e
    else:
        try: text=data.decode("utf-8-sig")
        except UnicodeDecodeError as e: raise ValueError("Text document must be UTF-8 encoded.") from e
    return re.sub(r"\n{3,}","\n\n",re.sub(r"[ \t]+"," ",text.replace("\x00"," "))).strip()
def chunk_text(text,size=800,overlap=120):
    if size<=0 or overlap<0 or overlap>=size: raise ValueError("Chunk overlap must be nonnegative and smaller than chunk size.")
    words=text.split(); chunks=[]; start=0
    while start<len(words):
        chunk=" ".join(words[start:start+size])
        if chunk: chunks.append(chunk)
        if start+size>=len(words): break
        start+=size-overlap
    return chunks
