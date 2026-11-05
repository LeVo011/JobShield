"""Evidence-grounded local explanation with an optional external LLM adapter."""
import re
from src.config import LLM_API_KEY,LLM_ENDPOINT,LLM_MODEL

RELEVANCE_THRESHOLD=0.03
TOKEN_RE=re.compile(r"[a-z0-9]+",re.I)
STOP_WORDS={"the","and","for","with","from","that","this","are","you","your","not","but","can","will","should","when","what","how","does","into","have","has","its","our","they","their","was","were","been","being","about","after","before","through","using","only","also","than","then","them","there","here","message","recruiter","job","role","to","in","is","of","a","an","or","on","at","as","by","be","it","we","us","do","did","does","their","they","he","she","i","me","my"}
TERM_SYNONYMS={"fee":{"payment","charge","deposit","registration"},"fees":{"payments","charges","deposits"},"registration":{"fee","enrollment","recruitment"},"otp":{"password","credential","verification"},"password":{"otp","credential","login"},"aadhaar":{"identity","identity-proof"},"bank":{"banking","account","financial"},"link":{"url","website","domain"},"urgent":{"urgency","immediate","pressure"}}

def _tokens(text):
    return {t.lower() for t in TOKEN_RE.findall(str(text or "")) if len(t)>1 and t.lower() not in STOP_WORDS}

def _expanded_tokens(text):
    terms=_tokens(text)
    for term in tuple(terms): terms.update(TERM_SYNONYMS.get(term,set()))
    return terms

def _relevant_passages(message,indicators,passages):
    """Keep ranked passages above a local floor with topical lexical support."""
    query_terms=_expanded_tokens(message)|_expanded_tokens(" ".join(f"{i.get('name','')} {i.get('span','')}" for i in indicators or []))
    relevant=[]
    for passage in passages or []:
        try: score=float(passage.get("similarity",0))
        except (TypeError,ValueError): score=0.0
        excerpt=str(passage.get("excerpt","")).strip()
        lexical_support=bool(query_terms&_expanded_tokens(excerpt))
        if score>=RELEVANCE_THRESHOLD and excerpt and lexical_support:
            relevant.append((passage,score))
    return relevant

def _summary_sentence(excerpt,message,indicators):
    sentences=[s.strip() for s in re.split(r"(?<=[.!?])\s+",str(excerpt or "").strip()) if s.strip()]
    if not sentences: return ""
    query=_tokens(message)|_tokens(" ".join(f"{i.get('name','')} {i.get('span','')}" for i in indicators or []))
    ranked=[]
    for index,sentence in enumerate(sentences):
        overlap=len(_tokens(sentence)&query)
        if overlap: ranked.append((overlap,-index,sentence))
    if ranked:
        ranked.sort(reverse=True)
        selected=[item[2] for item in ranked[:2]]
        # Keep source order in the displayed extractive summary.
        selected.sort(key=sentences.index)
        return " ".join(selected)
    return sentences[0]

def _source_label(passage,number):
    title=str(passage.get("title") or "").strip()
    source=str(passage.get("source") or "").strip()
    parts=[]
    if title: parts.append(title)
    if source: parts.append(source)
    citation=f"[{number}] "+" — ".join(parts) if parts else f"[{number}] Retrieved passage (source metadata unavailable)"
    date=passage.get("date")
    url=str(passage.get("url") or "").strip()
    if date: citation+=f" · date: {date}"
    if url: citation+=f" · URL: {url}"
    return citation

def _indicator_links(excerpt,indicators):
    evidence_terms=_expanded_tokens(excerpt)
    linked=[]
    for item in indicators or []:
        indicator_terms=_expanded_tokens(f"{item.get('name','')} {item.get('span','')}")
        if indicator_terms & evidence_terms: linked.append(item.get("name","Observed warning indicator"))
    return linked

def local_explanation(message,prediction,indicators,passages,diagnostic=""):
    """Build a transparent explanation from real rule outputs and retrieved text."""
    parts=["**Classifier prediction (model):** "+str(prediction.get("category","Unavailable"))+". "+str(prediction.get("explanation",""))]
    if indicators:
        observations="; ".join(f"{item.get('name','Indicator')} (severity: {item.get('severity','unspecified')}; matched text: “{item.get('span','')}”)" for item in indicators)
        parts.append("**Observed rule-based findings:** "+observations+". These are explicit pattern matches, not model explanations or proof of fraud.")
    else:
        parts.append("**Observed rule-based findings:** No configured warning pattern matched. This does not establish that the message is safe.")

    relevant=_relevant_passages(message,indicators,passages)
    if not relevant:
        parts.append("**Retrieved evidence:** Sufficient relevant supporting evidence was not found in the local knowledge base. No source citation is made.")
        parts.append("**Connection and interpretation:** With no relevant retrieved passage, the findings above should be treated as screening signals only. Verify the vacancy and recruiter independently.")
    else:
        evidence_lines=[]; connections=[]
        for number,(passage,score) in enumerate(relevant,1):
            summary=_summary_sentence(passage.get("excerpt",""),message,indicators)
            if not summary: continue
            citation=_source_label(passage,number)
            evidence_lines.append(f"- {summary} ({citation}; retrieval similarity {score:.3f}).")
            linked=_indicator_links(passage.get("excerpt",""),indicators)
            if linked:
                connections.append(f"{citation} has text overlap with the rule-based finding(s): {', '.join(linked)}. This is contextual guidance, not independent confirmation of this message.")
            else:
                connections.append(f"{citation} is general context; no direct wording overlap with the detected rule findings was found.")
        if evidence_lines:
            parts.append("**Retrieved evidence (extractive summary of local passages):**\n"+"\n".join(evidence_lines))
            parts.append("**How retrieved evidence relates to rule findings (inference):** "+" ".join(connections))
        else:
            parts.append("**Retrieved evidence:** The returned passages had no usable text; sufficient supporting evidence was not found. No source citation is made.")
            parts.append("**Connection and interpretation:** Treat the classifier and rule outputs as screening signals only; verify independently.")
    parts.append("**Overall interpretation:** JobShield has not verified recruiter identity, employer, vacancy, link, or source-page currency. False positives and false negatives are possible.")
    if diagnostic: parts.append("**Diagnostic:** "+diagnostic)
    relevant_metadata=[{**passage,"similarity":score} for passage,score in relevant]
    return {"mode":"Local template explanation","text":"\n\n".join(parts),"diagnostic":diagnostic or None,"relevant_passages":relevant_metadata}

def explain(message,prediction,indicators,passages):
    relevant=_relevant_passages(message,indicators,passages)
    if LLM_API_KEY and LLM_ENDPOINT and LLM_MODEL:
        try:
            import requests
            evidence="\n\n".join(f"[{n}] title={p.get('title') or 'metadata unavailable'}; source={p.get('source') or 'metadata unavailable'}; date={p.get('date') or 'unknown'}; url={p.get('url') or 'not supplied'}\nExcerpt: {p.get('excerpt','')}" for n,(p,_) in enumerate(relevant,1))
            prompt=("Explain the recruitment message using the supplied classifier output, explicit rule findings, and retrieved passages only. Separate retrieved evidence, rule-based observations, model prediction, and inference. Connect evidence to indicators only where supported by the excerpts. Cite sources using only supplied metadata; never invent sources, quotations, facts, or verification. If the evidence list is empty or insufficient, say so. Do not claim employer or recruiter verification.\n"
                    f"Message: {message}\nClassifier prediction: {prediction}\nRule findings: {indicators}\nRetrieved passages:\n{evidence or '[No sufficiently relevant passages retrieved]'}")
            response=requests.post(LLM_ENDPOINT,headers={"Authorization":f"Bearer {LLM_API_KEY}"},json={"model":LLM_MODEL,"messages":[{"role":"user","content":prompt}]},timeout=20)
            response.raise_for_status()
            content=response.json()["choices"][0]["message"]["content"]
            if not isinstance(content,str) or not content.strip(): raise ValueError("Empty LLM response")
            return {"text":content,"mode":"Configured external LLM","diagnostic":None,"relevant_passages":[{**p,"similarity":score} for p,score in relevant]}
        except Exception as exc:
            # Do not render exception text: it can contain endpoint URLs or request details.
            diagnostic=f"Configured LLM unavailable ({type(exc).__name__}); local explanation used."
            return local_explanation(message,prediction,indicators,passages,diagnostic)
    return local_explanation(message,prediction,indicators,passages)
