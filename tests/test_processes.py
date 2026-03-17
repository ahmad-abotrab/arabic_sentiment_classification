import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text

def test_clean_text_removes_url_and_mention():
    text = "مرحبا @user هذا رابط https://example.com"
    cleaned = clean_text(text)
    assert "@user" not in cleaned
    assert "http" not in cleaned

def test_clean_text_normalizes_arabic():
    text = "أحب البرمجة إلى أقصى حد"
    cleaned = clean_text(text)
    assert "أ" not in cleaned
    assert "إ" not in cleaned


if __name__ == "__main__":
    test_clean_text_removes_url_and_mention()
    test_clean_text_normalizes_arabic()
    print("tests passed")    