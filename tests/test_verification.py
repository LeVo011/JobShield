from src.verification import checklist
from src.indicators import detect_indicators
def test_checklist_tailored_and_reporting_link():
    items=checklist("Pay fee and send OTP",detect_indicators("Pay registration fee and send OTP"))
    assert any("Do not share" in x for x in items); assert any("cybercrime.gov.in" in x for x in items)
