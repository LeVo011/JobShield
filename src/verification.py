"""Generate a practical verification checklist from detected evidence."""
def checklist(text,indicators):
    result=["Locate the employer's official careers website independently and check that the vacancy exists.","Verify the recruiter's identity through a company contact found independently on the official website.","Do not pay fees or share OTPs, passwords, UPI PINs, or card security codes in response to an unsolicited message.","Inspect links and email domains carefully; navigate to the official site yourself instead of following an unexpected link."]
    names={i.get("name") for i in indicators}
    if "Payment request" in names: result.insert(0,"Pause any requested payment; independently ask the employer whether the charge is part of an official process.")
    if "Credential or OTP request" in names: result.insert(0,"Do not share the requested credential. If already disclosed, contact your bank or service provider promptly.")
    if "Sensitive identity or banking data" in names: result.insert(0,"Wait to share identity or banking documents until the employer and secure onboarding channel are verified.")
    result.append("If you suspect fraud, preserve messages and transaction details. Report through India's official portal: https://cybercrime.gov.in/")
    return list(dict.fromkeys(result))
