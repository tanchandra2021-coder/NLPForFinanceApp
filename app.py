############## STEP 1: Import libraries, load BERT model/tokenizer #############
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import plotly.express as px

@st.cache_resource  # Cache to load model/tokenizer only once per session
def load_model_and_tokenizer():
    # Use a standard pretrained BERT model
    model_name = "bert-base-uncased"  # Hugging Face model
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Set model to evaluation mode and move to GPU if available
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

########################### STEP 2: Helper Functions ###########################
def prepare_input(text):
    # Tokenize using the tokenizer, adding special tokens automatically
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']

def predict(text):
    input_ids, attention_mask = prepare_input(text)
    with torch.no_grad():
        outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device))
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy().flatten()
    return probabilities

########################## STEP 3: Streamlit interface #########################
st.title('Sentiment Analysis with BERT')
text = st.text_area("Enter the text to analyze below!")
button = st.button("Submit")

if button:
    if text:
        probabilities = predict(text)
        labels = ['Negative', 'Neutral', 'Positive']
        st.write("Sentiment Probabilities:")
        for label, prob in zip(labels, probabilities):
            st.write(f"{label}: {prob:.4f}")
    else:
        st.error("Please enter a text to analyze.")

