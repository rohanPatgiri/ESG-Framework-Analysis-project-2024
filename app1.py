import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize
import nltk

# Download NLTK punkt tokenizer
nltk.download('punkt')

# Function to tokenize sentences
def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Function to process PDF using PyMuPDF
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    num_pages = doc.page_count

    # Initialize an empty DataFrame to store the results
    df_combined = pd.DataFrame(columns=['Sentences'])

    for page_number in range(1, num_pages + 1):
        # Extract text from each page
        page = doc.load_page(page_number - 1)
        text = page.get_text()

        # Tokenize sentences using NLTK
        sentences = tokenize_sentences(text)

        # Create a DataFrame with the sentences
        df_page = pd.DataFrame({'Sentences': sentences})

        # Concatenate the current page's DataFrame to the combined DataFrame
        df_combined = pd.concat([df_combined, df_page], ignore_index=True)

    return df_combined

# Streamlit app
def main():
    st.title("PDF to CSV Converter")

    # Upload PDF file through Streamlit
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Process the PDF and get the combined DataFrame
        df_result = process_pdf(uploaded_file)

        # Display the combined DataFrame
        st.write(df_result.head())

        # Download CSV button
        st.download_button(
            label="Download CSV",
            data=df_result.to_csv(index=False).encode(),
            file_name="output_sentences_dataframe.csv",
            key="download_button"
        )

if __name__ == "__main__":
    main()

