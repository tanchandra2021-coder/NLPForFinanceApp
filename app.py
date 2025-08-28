# ↖️ Run this cell every time you make any changes!

############## STEP 1: Import libraries, load BERT model/tokenizer #############

# Imports
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import plotly.express as px

@st.cache_resource  # Cache to load model/tokenizer only once per session
def load_model_and_tokenizer():

    # Load saved model and tokenizer
    model_path = "NLP_For_Finance_BERT_Model" # <- Change if needed!
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Set model to evaluation mode (for predicting), and move to GPU if possible
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model, tokenizer, device

# Call the function to get cached model, tokenizer, and device
model, tokenizer, device = load_model_and_tokenizer()


########################### STEP 2: Helper Functions ###########################

### Preprocessing Function
def prepare_input(text):
  # Add special tokens
  sentence_with_tokens = "[CLS] " + text + " [SEP]"

  # Tokenize sentence
  tokenized_text = tokenizer.tokenize(sentence_with_tokens)

  # Convert tokens to IDs
  input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)

  # Pad the input IDs
  input_ids = pad_sequences([input_ids],
                              maxlen=128,
                              dtype="long",
                              truncating="post",
                              padding="post")[0]

  # Create attention masks
  attention_mask = [float(i>0) for i in input_ids]
  return torch.tensor([input_ids]), torch.tensor([attention_mask])


### Prediction Function
def predict(text):
  # Use our processing function on the input text
  input_ids, attention_mask = prepare_input(text)

  # Pass the processed data to the model
  with torch.no_grad():
      outputs = model(input_ids.to(device), token_type_ids=None, attention_mask=attention_mask.to(device))

  # Conver the output logits to probabilities and return!
  logits = outputs[0]
  probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy().flatten()
  return probabilities


########################## STEP 3: Streamlit interface #########################

### Interface
st.title('Sentiment Analysis with BERT')
text = st.text_area("Enter the text to analyze below!")
button = st.button("Submit")
### Button logic
if button:
  if text:
    ### Get Model's Prediction

    # YOUR CODE HERE: Replace the None with the model's prediction!
    probabilities = predict(text)
    ### END CODE HERE

    # Pair the labels with the probabilities in a dictionary
    labels = ['Negative', 'Neutral', 'Positive']

    # Print out the results on the Streamlit site
    st.write("Sentiment Probabilities:")
    for label, prob in zip(labels, probabilities):
      st.write(f"{label}: {prob:.4f}")

    ### (Optional) Pie Chart Visualization (next section)


  else:
    st.error("Please enter a text to analyze.")

