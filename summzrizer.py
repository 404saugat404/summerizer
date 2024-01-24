import fitz  # PyMuPDF
from transformers import pipeline
from summarizer import Summarizer

def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as pdf_document:
            text = ""
            for page_number in range(pdf_document.page_count):
                page = pdf_document[page_number]
                text += page.get_text()
            return text
    except Exception as e:
        print(f"Error: {e}")
        return None

def abstractive_summarize(text):
    # Split the text into chunks of 512 tokens to stay within the model's limit
    chunk_size = 1000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Initialize an empty summary
    summary = ""

    # Summarize each chunk separately
    for chunk in chunks:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        chunk_summary = summarizer(chunk, max_length=130, min_length=100, do_sample=False)
        summary += chunk_summary[0]['summary_text']

    return summary

def user_interaction():
    # Prompt user for PDF file input
    pdf_path = input("Enter the path to the PDF file: ")

    # Check if the file exists
    try:
        with open(pdf_path):
            pass
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' does not exist.")
        return

    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path)

    if extracted_text:
        # Ask the user for the summarization type
        summarization_type = input("Choose summarization type (extractive/abstractive): ")

        # Perform summarization based on user's choice
        if summarization_type.lower() == 'extractive':
            summary = extractive_summarize(extracted_text)
        elif summarization_type.lower() == 'abstractive':
            summary = abstractive_summarize(extracted_text)
        else:
            print("Invalid summarization type. Please choose 'extractive' or 'abstractive'.")
            return

        # Print the summarized text
        print("Summary:")
        print(summary)

if __name__ == "__main__":
    user_interaction()
