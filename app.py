import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
 
# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="AI Text Summarizer", layout="wide")
 
# ✅ Load model
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
 
# ✅ Extract text from uploaded file
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()
 
    if file_type == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")
 
    elif file_type == "pdf":
        try:
            import pdfplumber
            with pdfplumber.open(uploaded_file) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages)
        except ImportError:
            st.error("pdfplumber is not installed. Run: pip install pdfplumber")
            return ""
 
    elif file_type == "docx":
        try:
            import docx
            from io import BytesIO
            doc = docx.Document(BytesIO(uploaded_file.read()))
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            st.error("python-docx is not installed. Run: pip install python-docx")
            return ""
 
    else:
        st.error(f"Unsupported file type: .{file_type}")
        return ""
 
# ✅ Summarization with length control
def summarize_text(text, length="Balanced"):
    length_params = {
        "Short":    {"max_length": 60,  "min_length": 20},
        "Balanced": {"max_length": 120, "min_length": 40},
        "Detailed": {"max_length": 200, "min_length": 80},
    }
    params = length_params.get(length, length_params["Balanced"])
 
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
 
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=params["max_length"],
        min_length=params["min_length"],
        num_beams=4,
        early_stopping=True
    )
 
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
 
# ✅ UI
st.title("🧠 AI Text Summarizer")
st.markdown("Paste text **or** upload a file (PDF, DOCX, TXT) to summarize.")
 
# --- Input Mode Toggle ---
input_mode = st.radio(
    "Choose input method:",
    ["✍️ Paste Text", "📁 Upload File"],
    horizontal=True
)
 
text_to_summarize = ""
 
if input_mode == "✍️ Paste Text":
    input_text = st.text_area("Paste your text here:", height=250)
    text_to_summarize = input_text
 
else:
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT"
    )
 
    if uploaded_file is not None:
        with st.spinner("Reading file..."):
            extracted = extract_text_from_file(uploaded_file)
 
        if extracted.strip():
            st.success(f"✅ Extracted {len(extracted.split())} words from **{uploaded_file.name}**")
            with st.expander("Preview extracted text"):
                st.write(extracted[:2000] + ("..." if len(extracted) > 2000 else ""))
            text_to_summarize = extracted
        else:
            st.warning("Could not extract text from the file. It may be empty or image-based.")
 
# --- Summary Length ---
summary_length = st.selectbox(
    "Summary Length",
    ["Short", "Balanced", "Detailed"]
)
 
# --- Summarize Button ---
if st.button("Summarize"):
    if not text_to_summarize.strip():
        st.warning("Please enter some text or upload a file!")
    else:
        cleaned = clean_text(text_to_summarize)
 
        with st.spinner("Summarizing..."):
            summary = summarize_text(cleaned, length=summary_length)
 
        st.subheader("📝 Summary")
        st.write(summary)
 
        st.download_button(
            label="⬇️ Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )