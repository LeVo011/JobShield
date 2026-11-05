"""Explainable warning rules kept separate from model prediction."""
import re
RULES=[
("Payment request","high",r"\b(registration|processing|security|training|interview|onboarding|placement)\s+(fee|charge|deposit|payment)s?\b|\b(pay|transfer|deposit|send)\s+(a|the|your)?\s*(fee|money|deposit|charge)\b","Recruitment-related payment demands deserve close scrutiny. Independently confirm any legitimate fee policy; do not pay in response to an unsolicited message."),
("Credential or OTP request","critical",r"\b(otp|one[- ]time password|password|upi pin|cvv|bank login|banking credentials|verification code)\b","Employers should not need your password, OTP, PIN, or card security code to recruit you. Never share these secrets; contact your bank if already disclosed."),
("Sensitive identity or banking data","high",r"\b(aadhaar|aadhar|pan card|passport|bank account|account number|identity documents|identity proof)\b","Identity or banking details can enable misuse. Verify the employer and secure onboarding channel independently before sharing necessary documents."),
("Guaranteed or unusually high reward claim","medium",r"\b(guaranteed (selection|job|employment|placement)|no interview|\b\d+(?:\.\d+)?\s*(?:lakh|lakhs|crore|cr)\s*(?:per month|weekly|per week))\b","Guaranteed selection or extraordinary compensation without a credible process can be a warning sign. Compare the vacancy with the employer's official listing."),
("Urgency or pressure","medium",r"\b(immediately|within \d+ minutes|act now|urgent|expires in|last chance|lose this opportunity|or lose|threaten|legal action)\b","Pressure can discourage careful checks. Pause and verify using independently found company contact details."),
("Unverified contact or unusual channel","low",r"\b(whatsapp|telegram|gift cards?|cryptocurrency|crypto|personal wallet|private account|do not contact|don't contact)\b","An unusual channel or instruction to avoid the employer warrants verification. Some legitimate recruiters use messaging apps, so check through the official company contact."),
("Link requires scrutiny","medium",r"https?://[^\s]+|\bwww\.[^\s]+|\b(bit\.ly|tinyurl|t\.co)/[^\s]+","A link is present, but this check does not verify its destination or ownership. Do not enter credentials; navigate to the official site independently.")]
NEXT={"critical":"Do not share the secret. If already shared, contact the relevant bank/service immediately.","high":"Pause before sending money or documents; verify the vacancy and recruiter through official contacts.","medium":"Verify the role and claims through the employer's independently located careers page.","low":"Confirm the channel and identity using contact details published by the employer."}
def detect_indicators(text:str):
    if not isinstance(text,str): return []
    found=[]
    for name,severity,pattern,why in RULES:
        m=re.search(pattern,text,re.I)
        if m: found.append({"name":name,"severity":severity,"span":m.group(0),"start":m.start(),"end":m.end(),"why":why,"next_step":NEXT[severity],"source":"Explicit rule-based pattern"})
    return found
