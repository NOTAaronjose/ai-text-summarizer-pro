import streamlit as st
from transformers import pipeline
import re

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="AI Text Summarizer", layout="wide")

# ✅ Load model (cached for performance)
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

summarizer = load_model()

# ✅ Clean input text
def clean_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ✅ Summarization function (with chunking for better results)
def summarize_text(text):
    paragraphs = text.split("\n")
    summaries = []

    for p in paragraphs:
        if len(p.strip()) > 30:
            result = summarizer(
                p,
                max_length=100,
                min_length=30,
                do_sample=False
            )
            summaries.append(result[0]['summary_text'])

    return " ".join(summaries)

# ✅ UI
st.title("🧠 AI Text Summarizer")

input_text = st.text_area("Paste your text here:", height=250)

summary_length = st.selectbox(
    "Summary Length",
    ["Short", "Balanced", "Detailed"]
)

if st.button("Summarize"):
    if input_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = clean_text(input_text)

        # Adjust summary length
        if summary_length == "Short":
            max_len, min_len = 60, 20
        elif summary_length == "Detailed":
            max_len, min_len = 150, 50
        else:
            max_len, min_len = 100, 30

        summary = summarize_text(cleaned)

        st.subheader("Summary")
        st.write(summary)