import streamlit as st
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "fashion_advisor_model"  # Path to your saved model
    model = FastLanguageModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Streamlit App
st.title("Fashion Advisor Chatbot")
st.write("Ask me for fashion advice!")

# User input
user_input = st.text_input("Enter your question:")

if user_input:
    # Prepare the prompt
    prompt = f"### Instruction:\nProvide fashion advice.\n\n### Input:\n{user_input}\n\n### Response:\n"
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate response
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Display response
    st.write("**Fashion Advisor:**", response.strip())
