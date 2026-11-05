"""JobShield Streamlit interface."""
import streamlit as st
import pandas as pd
from src.config import MAX_MESSAGE_CHARS,MAX_UPLOAD_MB
from src.classifier import load_model,predict
from src.indicators import detect_indicators
from src.retriever import LocalRetriever
from src.rag_pipeline import explain
from src.verification import checklist
from src.ui_state import next_analysis_generation,results_match_input,checklist_widget_key

st.set_page_config(page_title="JobShield",page_icon="🛡️",layout="wide")
st.markdown("<style>html,body,[class*='css']{font-family:Arial,sans-serif}h1,h2,h3{font-family:Arial,sans-serif}.hero{padding:1.4rem 1.8rem;border:1px solid #dce6ef;border-radius:18px;background:linear-gradient(115deg,#f4f8fc,#eef7f5);margin-bottom:1rem}.hero h1{color:#17324a!important;font-size:2.15rem;line-height:1.2;margin:0 0 .65rem}.hero p{color:#435a6e!important}.muted{color:#5d6b79}</style>",unsafe_allow_html=True)
@st.cache_resource
def get_retriever(): return LocalRetriever()
@st.cache_resource
def get_model(): return load_model()
EXAMPLES={
"1 · Internship registration fee":"Hello, I am hiring for fictional AsterByte Labs. Pay a refundable registration fee today to secure your internship. The offer expires soon.",
"2 · Recruiter asks for OTP":"Your application is selected at fictional CedarCloud. Send the OTP received on your phone and your password so we can issue your appointment letter.",
"3 · Guaranteed high salary":"Fictional BrightRoute is offering 8 lakh per month with guaranteed selection and no interview. Act now to claim your role.",
"4 · Ordinary interview invitation":"Thank you for applying to fictional Northstar Labs. We would like to schedule a 30-minute interview for the junior analyst role. Please choose a time through our careers page.",
"5 · Identity and banking details":"A recruiter from fictional MapleWorks asks you to send Aadhaar, PAN and bank account details through an unverified messaging account before sharing the role description.",
"6 · Short ambiguous message":"Hi, are you interested in a job?"}
st.sidebar.markdown("# 🛡️ JobShield")
page=st.sidebar.radio("Workspace",["Analyze","Knowledge base","Evaluation","About & limitations"],label_visibility="collapsed")
st.sidebar.caption("Local-first screening assistant · India")
st.sidebar.info("A warning signal is not proof of fraud. A lower-risk result does not establish legitimacy.")
def run_analysis(message):
    model=get_model(); pred=predict(message,model); indicators=detect_indicators(message); retriever=get_retriever(); retrieved=retriever.search(message+" "+" ".join(i["name"] for i in indicators),4); explanation=explain(message,pred,indicators,retrieved); passages=explanation.get("relevant_passages",[])
    generation=next_analysis_generation(st.session_state.get("analysis_generation",0))
    st.session_state.analysis_generation=generation
    st.session_state.analysis={"message":message,"generation":generation,"prediction":pred,"indicators":indicators,"passages":passages,"explanation":explanation,"checklist":checklist(message,indicators)}
def show_analysis(a):
    p=a["prediction"]; left,right=st.columns([1,2])
    with left:
        st.subheader("Assessment"); st.metric("Model category",p["category"])
        if p["score"] is not None: st.progress(min(max(p["score"],0),1),text=f"Uncalibrated model score: {p['score']:.2f}")
        st.caption("The score is not calibrated and is not a probability of fraud."); st.info(p["explanation"])
    with right:
        st.subheader("Observed warning indicators")
        if not a["indicators"]: st.success("No configured warning pattern detected. This does not prove the message is safe.")
        for i in a["indicators"]:
            with st.container(border=True):
                st.markdown(f"**{i['name']}** · `{i['severity'].upper()}`\n\n> {i['span']}")
                st.write(i["why"]); st.caption("Practical step: "+i["next_step"])
    st.subheader("Evidence and explanation"); st.caption("Explanation mode: "+a["explanation"]["mode"]); st.write(a["explanation"]["text"])
    if a["passages"]:
        for n,x in enumerate(a["passages"],1):
            with st.expander(f"[{n}] {x['title']} — {x.get('source','Unknown source')} · similarity ranking {x['similarity']:.2f}"):
                st.write(x["excerpt"])
                if x.get("date"): st.caption("Publication date: "+str(x["date"]))
                if x.get("url"): st.markdown("Source: [Open source page]("+x["url"]+")")
    else: st.info("Sufficient supporting evidence was not found in the local knowledge base.")
    st.subheader("Verification checklist")
    for n,step in enumerate(a["checklist"]): st.checkbox(step,key=checklist_widget_key(a.get("generation",0),n))
    st.warning("JobShield does not verify recruiter identity, vacancy status, domains, or employer claims in real time. False positives and false negatives are possible.")
if page=="Analyze":
    st.markdown("<div class='hero'><h1>Recruitment safety, with evidence.</h1><p class='muted'>Screen job offers, internship ads, recruiter messages, and emails for warning signs and practical next steps.</p></div>",unsafe_allow_html=True)
    st.write("JobShield combines text classification, explicit warning rules, and a local guidance library. Your message is processed locally by default.")
    def _fill_example():
        selected=st.session_state.get("example_choice","Choose an example…")
        st.session_state["recruitment_message"]=EXAMPLES.get(selected,"")
    if "recruitment_message" not in st.session_state: st.session_state.recruitment_message=""
    st.selectbox("Illustrative example messages",["Choose an example…",*EXAMPLES],key="example_choice",on_change=_fill_example)
    message=st.text_area("Recruitment message",height=210,max_chars=MAX_MESSAGE_CHARS,key="recruitment_message",placeholder="Paste the message you'd like to assess…")
    c1,c2=st.columns([1,4])
    with c1: analyze=st.button("Analyze message",type="primary",use_container_width=True)
    with c2: st.caption(f"{len(message):,} / {MAX_MESSAGE_CHARS:,} characters · Avoid pasting unrelated personal information.")
    if analyze:
        if not message.strip(): st.error("Enter a message before analyzing.")
        else:
            with st.spinner("Analyzing text and retrieving local guidance…"):
                try: run_analysis(message); st.success("Analysis ready.")
                except Exception as exc: st.error(f"Analysis failed ({type(exc).__name__}). Check the input and local classifier/index, then retry.")
    analysis=st.session_state.get("analysis")
    if analysis:
        if results_match_input(analysis,message): show_analysis(analysis)
        else: st.info("The message changed. Run Analyze message to replace the previous results.")
elif page=="Knowledge base":
    st.title("Knowledge base"); st.write("Add local PDF, TXT, or Markdown references to the starter safety guidance.")
    uploaded=st.file_uploader("Add a document (maximum 10 MB)",type=["pdf","txt","md","markdown"])
    if st.button("Index uploaded document",disabled=uploaded is None):
        try:
            if uploaded.size>MAX_UPLOAD_MB*1024*1024: raise ValueError("Document exceeds 10 MB.")
            count=get_retriever().add_document(uploaded.name,uploaded.getvalue()); st.success(f"Indexed {count} chunk(s) from {uploaded.name}.")
        except Exception as exc: st.error(f"Ingestion failed ({type(exc).__name__}). Check the file type, 10 MB limit, encoding, and whether text can be extracted.")
    if st.button("Rebuild local index"):
        try: get_retriever().rebuild(); st.success("Knowledge-base index rebuilt.")
        except Exception as exc: st.error(f"Index rebuild failed ({type(exc).__name__}). Check saved documents and local storage, then retry.")
    docs=get_retriever().documents()
    if docs: st.dataframe(pd.DataFrame(docs),use_container_width=True,hide_index=True)
    else: st.info("No documents indexed yet.")
    retriever=get_retriever()
    user_docs=[d["filename"] for d in docs if d.get("filename")]
    if user_docs:
        selected_doc=st.selectbox("Remove an uploaded document",user_docs,key="delete_document_choice")
        if st.button("Delete selected document"):
            try:
                if retriever.delete_document(selected_doc): st.success("Document removed and index rebuilt."); st.rerun()
                else: st.info("That document is no longer present.")
            except Exception as exc: st.error(f"Document deletion failed ({type(exc).__name__}).")
    if retriever.diagnostic: st.info(retriever.diagnostic)
    for message in retriever.document_diagnostics: st.warning(message)
    st.caption(f"Retrieval backend: {retriever.embedder.backend}. {retriever.embedder.diagnostic or 'Local embedding model available.'} Similarity is a ranking, not a probability.")
elif page=="Evaluation":
    st.title("Evaluation dashboard")
    try: meta=get_model()["metadata"]
    except Exception: st.error("Model unavailable. Use the training command in the README and retry."); st.stop()
    st.warning(meta["warning"]); st.caption(f"Dataset: {meta['dataset']['source']} · {meta['dataset']['kind']} · random seed {meta['seed']}")
    st.caption("Classifier artifact scikit-learn version: "+str(meta.get("scikit_learn_version","unknown")))
    a,b,c=st.columns(3); a.metric("Dataset rows",meta["dataset_size"]); b.metric("Training rows",meta["train_size"]); c.metric("Held-out test rows",meta["test_size"])
    st.subheader("Class distribution")
    distribution=pd.DataFrame({"Messages":[meta["class_counts"]["lower_risk_indicators"],meta["class_counts"]["potentially_suspicious"]]},index=["Lower-risk","Potentially suspicious"])
    st.bar_chart(distribution)
    labels=[("Lower-risk indicators","lower_risk_indicators"),("Potentially suspicious","potentially_suspicious")]
    st.subheader("Held-out per-class metrics"); st.dataframe(pd.DataFrame([{"Class":name,"Precision":meta["metrics"][key]["precision"],"Recall":meta["metrics"][key]["recall"],"F1":meta["metrics"][key]["f1-score"],"Support":meta["metrics"][key]["support"]} for name,key in labels]),hide_index=True,use_container_width=True)
    st.write("Macro averages",{k:round(meta["metrics"]["macro avg"][k],3) for k in ["precision","recall","f1-score"]}); st.write("Confusion matrix: rows=true [lower-risk, suspicious], columns=predicted [lower-risk, suspicious]")
    st.dataframe(pd.DataFrame(meta["confusion_matrix"],index=["True lower-risk","True suspicious"],columns=["Predicted lower-risk","Predicted suspicious"]),use_container_width=True)
    st.info("No simple baseline comparison is shown. Held-out metrics are not production estimates; synthetic fallback metrics only measure the small illustrative set.")
    st.code("python scripts\\evaluate_model.py --data data\\emscad.csv")
else:
    st.title("About JobShield"); st.subheader("Architecture")
    st.markdown("- **Classifier:** scikit-learn TF-IDF and Logistic Regression with deterministic held-out split.\n- **Indicators:** explicit pattern rules with matched spans and actions.\n- **Retrieval:** cached Sentence Transformers if present; TF-IDF cosine fallback.\n- **Explanation:** evidence-aware local template by default; optional configured OpenAI-compatible LLM.\n- **Documents:** PyMuPDF for PDFs; UTF-8 TXT and Markdown.")
    st.subheader("Intended use and limitations"); st.write("JobShield is an educational screening assistant for Indian students, graduates, and job seekers. It identifies warning signs; it does not establish that a message, recruiter, domain, vacancy, or employer is genuine or fraudulent. Models can miss scams or flag legitimate messages. Do not make legal, financial, or employment decisions from the score alone.")
    st.subheader("Privacy"); st.write("Messages are processed locally and are not logged. User-added documents remain in the local knowledge-base folder and are excluded from Git. If all optional LLM settings are configured, message text and findings may be sent to that endpoint; review its data handling before enabling. No API key is needed for core features.")
    st.subheader("Dataset"); st.write("No EMSCAD data is bundled. Without a user-supplied CSV the app trains on hand-authored synthetic examples, labelled demonstration-only. See README and data/README.md for schema and licensing caveat.")
    st.markdown("For suspected cybercrime in India, use the [official National Cyber Crime Reporting Portal](https://cybercrime.gov.in/).")
