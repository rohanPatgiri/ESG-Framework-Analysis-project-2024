import pandas as pd
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch

# Load the ESG scoring model from Hugging Face
model_name = "yiyanghkust/finbert-esg"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
esg_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True)

# Function to get E, S, G scores for a sentence
def get_esg_scores(sentence):
    try:
        scores = esg_pipeline(sentence)
        return {score['label']: score['score'] for score in scores}
    except RuntimeError as e:
        print(f"Error for sentence number : {row}. Skipping row. Error details: {e}")
        return None

# Function to calculate average scores from a DataFrame
def calculate_average_scores(df):
    average_scores = {}
    for label in ['Environmental', 'Social', 'Governance']:
        column_name = f'{label}_Score'
        non_blank_entries = df[column_name].count()
        sum_scores = df[column_name].sum(skipna=True)
        average_score = sum_scores / non_blank_entries if non_blank_entries > 0 else 0
        average_scores[label] = average_score
    return average_scores

# Function to process CSV and calculate average scores
def process_csv_and_calculate_average(csv_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Iterate over each row and update E, S, G scores
    for index, row in df.iterrows():
        sentence = row['Sentences']  # Replace with the actual column name
        scores = get_esg_scores(sentence)
        
        if scores is not None:
            for label, score in scores.items():
                df.at[index, f'{label}_Score'] = score

    # Calculate and return average scores
    return calculate_average_scores(df)

# Streamlit App
def main():
    st.title("ESG Scores Dashboard")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Process and calculate average scores
        average_scores = process_csv_and_calculate_average(uploaded_file)

        # Display average scores
        st.write("Average Environmental Score:", average_scores['Environmental'])
        st.write("Average Social Score:", average_scores['Social'])
        st.write("Average Governance Score:", average_scores['Governance'])

if __name__ == "__main__":
    main()
