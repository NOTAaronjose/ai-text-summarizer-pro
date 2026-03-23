🧠 AI Text Summarizer

AI-Powered Text Summarization App built with Streamlit and HuggingFace Transformers.
Paste text or upload a document (PDF, DOCX, TXT) and get a concise summary in seconds. Supports long documents with chunked summarization.

Features
✍️ Paste Text – Enter any text to summarize instantly.
📁 File Upload – Upload PDF, DOCX, TXT files.
📄 Long Document Support – Automatically splits large files into chunks for complete summarization.
⚡ Chunked Summarization – Ensures accurate summaries even for long documents.
⏳ Progress Feedback – Shows progress bar and status messages for multi-chunk summarization.
🎨 Custom UI – Dark theme, clean hero section, summary card, and styled buttons.
⬇️ Download Summary – Save summaries as .txt.
Demo

Live App: [AI Text Summarizer](https://ai-text-summarizer-pro-dyxyaxrqo83yfgbr5vrdfa.streamlit.app/)

Installation

Clone the repository:

git clone https://github.com/NOTAaronjose/ai-text-summarizer-pro.git
cd ai-text-summarizer-pro

Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Usage

Run the app locally:

streamlit run app.py
Paste Text: Type or paste your text into the input area.
Upload File: Choose a PDF, DOCX, or TXT file.
Select Summary Length: Short / Balanced / Detailed.
Click “Summarize” to get your summary.
Download the summary using the button below the result.
Project Structure
ai-text-summarizer-pro/
│
├─ app.py               # Main Streamlit app
├─ requirements.txt     # Python dependencies
├─ README.md            # Documentation
├─ assets/              # Optional: screenshots, demo images, icons
└─ ...
Dependencies
streamlit – Web interface
transformers – HuggingFace models for summarization
torch – Deep learning backend
pdfplumber – Extract text from PDFs
python-docx – Extract text from DOCX
regex – For text processing

Optional (for OCR on scanned PDFs):

pytesseract
Pillow
Future Improvements
Add OCR support for scanned PDFs.
Export summaries as PDF or bullet points.
Add multi-language support using mBART or similar models.
Save summarization history for user sessions.
License

MIT License – free to use, modify, and share.
