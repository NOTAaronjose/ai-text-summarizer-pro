import streamlit as st
from transformers import pipeline
import re

# ✅ MUST be first Streamlit command
st.set_page_config(page_title="AI Text Summarizer", layout="wide")

# ✅ Load model (fixed pipeline)
@st.cache_resource
def load_model():
    return pipeline(
        task="text2text-generation",  # 🔥 FIXED (was "summarization")
        model="sshleifer/distilbart-cnn-12-6"
    )

summarizer = load_model()

# ✅ Clean input text
def clean_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ✅ Summarization with chunking
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
            summaries.append(result[0]['generated_text'])  # 🔥 changed key

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

        summary = summarize_text(cleaned)

        st.subheader("Summary")
        st.write(summary)