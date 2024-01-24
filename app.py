
import streamlit as st
from transformers import pipeline
from summarizer import Summarizer
import fitz
from io import BytesIO

def extract_text_from_pdf(pdf_file):
    try:
        with fitz.open("pdf", pdf_file.read()) as pdf_document:
            text = ""
            for page_number in range(pdf_document.page_count):
                page = pdf_document[page_number]
                text += page.get_text()
            return text
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def extractive_summarize(text):
    summarizer = Summarizer()
    return summarizer(text)

def abstractive_summarize(text):
    # Split the text into chunks of 512 tokens to stay within the model's limit
    chunk_size = 512
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Initialize an empty summary
    summary = ""

    # Summarize each chunk separately
    for chunk in chunks:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        chunk_summary = summarizer(chunk, max_length=130, min_length=100, do_sample=False)

        # Check if chunk_summary is not empty
        if chunk_summary:
            summary += chunk_summary[0]['summary_text']

    return summary

def main():
    st.title("PDF Summarizer")

    # File upload section with drag-and-drop
    pdf_file = st.file_uploader("Drag and drop a PDF file here", type=["pdf"], key="fileUploader")

    if pdf_file is not None:
        # Extract text from the uploaded PDF file
        extracted_text = extract_text_from_pdf(pdf_file)

        if extracted_text:
            # Summarization type selection
            summarization_type = st.radio("Choose summarization type", ('Extractive', 'Abstractive'))

            # Perform summarization based on user's choice
            if summarization_type == 'Extractive':
                summary = extractive_summarize(extracted_text)
            elif summarization_type == 'Abstractive':
                summary = abstractive_summarize(extracted_text)

            # Display the summarized text
            st.subheader("Summary:")
            st.write(summary)

if __name__ == "__main__":
    main()

