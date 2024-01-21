import streamlit as st
import base64
import PyPDF2
from PyPDF2 import PdfReader
import pandas as pd
from nltk.tokenize import sent_tokenize

def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return sentences
def create_download_link(df, link_text, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert DataFrame to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def process_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    # Initialize an empty DataFrame to store the results
    df_combined = pd.DataFrame(columns=['Sentences'])

    for page_number in range(1, num_pages + 1):
        # Extract text from each page
        page = reader.pages[page_number - 1]
        extracted_text = page.extract_text()

        # Tokenize sentences using NLTK
        sentences = tokenize_sentences(extracted_text)

        # Create a DataFrame with the sentences
        df_page = pd.DataFrame({'Sentences': sentences})

        # Concatenate the current page's DataFrame to the combined DataFrame
        df_combined = pd.concat([df_combined, df_page], ignore_index=True)

    return df_combined

def main():
    st.title("PDF to CSV Converter (for CAS Project)")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Process PDF and get the combined DataFrame
        df_result = process_pdf(uploaded_file)

        # Download link for the CSV file
        csv_link = create_download_link(df_result, "Download CSV", "sentences_dataframe_combined.csv")
        st.markdown(csv_link, unsafe_allow_html=True)

        # Display a sample of the DataFrame
        st.dataframe(df_result.head())

if __name__ == "__main__":
    main()
