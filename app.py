import streamlit as st
from transformers import pipeline
import re

@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_model()

def clean_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def to_bullets(summary, n=4):
    sentences = re.split(r'(?<=[.!?]) +', summary)
    bullets = []
    for s in sentences:
        if len(s.split()) > 6:
            bullets.append(s.strip())
    return bullets[:n]

def chunked_summarize(text, max_len, min_len):
    words = text.split()
    chunks = [" ".join(words[i:i+400]) for i in range(0, len(words), 400)]
    summaries = []
    for chunk in chunks:
        result = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
        summaries.append(result[0]["summary_text"])
    combined = " ".join(summaries)
    final = summarizer(combined, max_length=max_len, min_length=min_len, do_sample=False)
    return final[0]["summary_text"]

def generate_title(text):
    result = summarizer(text[:300], max_length=20, min_length=5, do_sample=False)
    return result[0]["summary_text"]

st.set_page_config(page_title="AI Text Summarizer", layout="wide")
st.title("🧠 AI Text Summarizer")

text_input = st.text_area("Paste your text here:", height=250)

mode = st.selectbox("Summary Length", ["Short", "Balanced", "Detailed"])
bullet_mode = st.checkbox("Show as bullet points", value=True)

if st.button("Summarize"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text_input)

        if mode == "Short":
            max_len, min_len = 100, 30
        elif mode == "Detailed":
            max_len, min_len = 300, 120
        else:
            max_len, min_len = 180, 60

        summary = chunked_summarize(cleaned, max_len, min_len)
        title = generate_title(cleaned)

        st.subheader(title)

        if bullet_mode:
            for i, b in enumerate(to_bullets(summary), 1):
                st.write(f"{i}. {b}")
        else:
            st.write(summary)