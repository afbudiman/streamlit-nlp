from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import streamlit as st

def main():
    txt = st.text_area(label='Put your review here')

    model_name = "aychang/roberta-base-imdb"

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    nlp = pipeline(model=model, tokenizer=tokenizer)

    results = nlp(txt)

    st.write('Sentiment: ', results)
    

if __name__ == "__main__":
    st.set_page_config(page_title="Sentiment Analysis")
    st.title("Sentiment Analysis")
    main()
