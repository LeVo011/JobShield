from src.indicators import detect_indicators
def test_payment_and_otp_indicators_have_spans_and_actions():
    found=detect_indicators("Pay registration fee and send OTP now")
    assert {x["name"] for x in found}>={"Payment request","Credential or OTP request"}
    assert all(x["span"] and x["next_step"] for x in found)
def test_ordinary_copy_not_flagged():
    assert detect_indicators("Thank you for applying. We invite you to an interview.")==[]
