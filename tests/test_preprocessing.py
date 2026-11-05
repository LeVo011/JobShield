import pytest
from src.preprocessing import normalize_text
def test_normalizes_unicode_whitespace_and_url():
    assert normalize_text("  HI https://example.com/a  ") == "hi url"
def test_rejects_non_text():
    with pytest.raises(TypeError): normalize_text(None)
