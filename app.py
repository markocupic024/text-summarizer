import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import newspaper
import re

@st.cache_resource
def load_model():
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

def summarize(text, max_length=1024, min_length=50, num_beams=4, length_penalty=2.0, summary_length=100):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=num_beams, length_penalty=length_penalty,
                                 max_length=max_length, min_length=min_length, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    sentences = re.split(r'(?<=[.!?]) +', summary)
    truncated_summary = ""
    word_count = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if word_count + len(sentence_words) > summary_length:
            break
        truncated_summary += sentence + " "
        word_count += len(sentence_words)
    
    return truncated_summary.strip()

def fetch_article(url):
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        st.error(f"Error fetching article: {e}")
        return None

def main():
    st.title("Text Summarizer")
    st.write("Summarize long texts or articles using AI. Choose to input text directly or provide a URL.")

    input_type = st.radio("Select input type:", ("Text", "URL"))

    if input_type == "Text":
        text = st.text_area("Enter the text to summarize:", height=200)
    else:
        url = st.text_input("Enter the URL of the article:")
        if url:
            with st.spinner("Fetching article..."):
                text = fetch_article(url)
            if text:
                st.write(f"Article fetched successfully! Length: {len(text)} characters.")
        else:
            text = None

    summary_length = st.slider("Desired summary length (in words):", min_value=50, max_value=300, value=100, step=10)

    if st.button("Summarize"):
        if text:
            with st.spinner("Summarizing..."):
                summary = summarize(text, summary_length=summary_length)
            st.subheader("Summary")
            st.write(summary)
        else:
            st.error("Please provide valid input.")

if __name__ == "__main__":
    main()