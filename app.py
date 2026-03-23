import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="AI Text Summarizer", layout="wide")

# ✅ Load model (NO pipeline = no KeyError)
@st.cache_resource
def load_model():
    model_name = "sshleifer/distilbart-cnn-12-6"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_model()

# ✅ Clean input text
def clean_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ✅ Summarization function
def summarize_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=120,
        min_length=40,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

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

        with st.spinner("Summarizing..."):
            summary = summarize_text(cleaned)

        st.subheader("Summary")
        st.write(summary)