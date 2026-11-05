"""Conservative text normalization."""
import re, unicodedata
def normalize_text(text: str) -> str:
    if not isinstance(text,str): raise TypeError("Message must be text.")
    text=unicodedata.normalize("NFKC",text).replace("\x00"," ")
    text=re.sub(r"https?://\S+|www\.\S+"," URL ",text,flags=re.I)
    text=re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"," EMAIL ",text)
    return re.sub(r"\s+"," ",text).strip().lower()
