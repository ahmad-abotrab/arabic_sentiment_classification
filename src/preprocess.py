import re
import pandas as pd

ARABIC_DIACRITICS = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
TATWEEL_PATTERN = re.compile(r'ـ+')
MULTISPACE_PATTERN = re.compile(r'\s+')

def normalize_arabic(text: str) -> str:
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    return text

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()
    text = URL_PATTERN.sub('', text)
    text = MENTION_PATTERN.sub('', text)
    text = TATWEEL_PATTERN.sub('', text)
    text = ARABIC_DIACRITICS.sub('', text)
    text = normalize_arabic(text)
    text = MULTISPACE_PATTERN.sub(' ', text)
    return text.strip()

def clean_dataframe(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    df = df[[text_col, label_col]].copy()
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]
    df = df.drop_duplicates(subset=[text_col])

    df[text_col] = df[text_col].apply(clean_text)
    df = df[df[text_col] != ""]
    df = df.drop_duplicates(subset=[text_col])

    df = df.rename(columns={text_col: "text", label_col: "label"})
    return df.reset_index(drop=True)