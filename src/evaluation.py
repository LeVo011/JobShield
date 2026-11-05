"""Held-out metric generation and 30 actual demo-message checks."""
import json
from pathlib import Path
from src.classifier import train_model,load_model,predict
from src.data_loader import SUSPICIOUS,LOWER_RISK
from src.config import EVAL_PATH,MODEL_PATH
SCENARIO_CASES=[
 ("Fictional Orion internship says send a refundable enrollment fee before your first day.",1),
 ("A recruiter asks me to read out the OTP sent to my phone to confirm an offer.",1),
 ("No interview needed: earn 4 lakh every month, selection guaranteed if you act today.",1),
 ("Send your UPI PIN and bank password so payroll can activate your new role.",1),
 ("The recruiter wants payment in gift cards to reserve a private interview slot.",1),
 ("Urgent vacancy: pay laptop delivery charges to this personal wallet immediately.",1),
 ("You are hired without applying. Share your Aadhaar and account credentials in chat.",1),
 ("Pay a background check deposit now or we will cancel your guaranteed placement.",1),
 ("The recruiter says never call the company and asks for my card CVV.",1),
 ("A message promises 6 lakh weekly and requires a processing charge for selection.",1),
 ("Click a shortened link and enter your banking login to receive the appointment letter.",1),
 ("A recruiter threatens legal action unless I send my one-time password.",1),
 ("Buy the required training kit from the recruiter before they disclose the employer.",1),
 ("Transfer an interview fee to a cryptocurrency wallet for guaranteed hiring.",1),
 ("The offer expires in minutes; send your identity proof and account number by chat.",1),
 ("I applied for the graduate role and received an invitation to choose an interview time.",0),
 ("The hiring team will review my resume and contact shortlisted candidates next week.",0),
 ("The company careers page lists the position and asks applicants to submit a portfolio.",0),
 ("A recruiter scheduled a technical interview about a role I applied for last month.",0),
 ("The internship email describes the location, responsibilities, and stated monthly stipend.",0),
 ("I received a reminder for the video interview already in my calendar.",0),
 ("The campus office shared the employer's role details and interview schedule.",0),
 ("The recruiter requested my resume and three available times for a call.",0),
 ("My application status is available on the careers portal I visited independently.",0),
 ("The employer sent a written offer after two interviews and provided a contact number on its website.",0),
 ("A hiring coordinator answered questions about the role and its published salary band.",0),
 ("The interview panel asked me to prepare a short design exercise for the role.",0),
 ("I received confirmation that my application was submitted successfully.",0),
 ("The recruiter invited me to discuss the advertised software internship by phone.",0),
 ("A recruiter sent me a message.",None),
]
def evaluate(data_path=None,output_path=EVAL_PATH):
    result=train_model(data_path=data_path,model_path=MODEL_PATH); Path(output_path).parent.mkdir(parents=True,exist_ok=True); Path(output_path).write_text(json.dumps(result,indent=2),encoding="utf-8"); return result
def assess_30(model_path=MODEL_PATH,output_path=None):
    bundle=load_model(model_path); rows=[]
    for text,expected in SCENARIO_CASES:
        p=predict(text,bundle); actual=None if p["score"] is None else int(p["score"]>=.5)
        expected_category="Uncertain assessment" if expected is None else ("Potentially suspicious" if expected else "Lower-risk indicators detected")
        rows.append({"input":text,"expected":expected_category,"actual":p["category"],"correct":p["category"]==expected_category,"score":p["score"]})
    if output_path: Path(output_path).write_text(json.dumps(rows,indent=2),encoding="utf-8")
    return rows
