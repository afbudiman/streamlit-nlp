# from sentiment_model import text_cleaning
# import pickle
import streamlit as st
from transformers import pipeline

model_id = "afbudiman/indobert-classification" 
classifier = pipeline("text-classification", model=model_id)


def main():
    txt = st.text_area(label='Put your review here')

    if st.button('Predict'):
        preds = classifier(txt, return_all_scores=False)
        output = preds[0]['label']
        probas = preds[0]['score']
        output_probability = "{:.2f}".format(probas)
        sentiments = {'LABEL_0': "POSITIVE", 'LABEL_1': "NEUTRAL", 'LABEL_2': "NEGATIVE"}
        st.write(f"It's {sentiments[output]} sentiment with {output_probability} probability")

# def main():
#     txt = st.text_area(label='Put your review here')

#     if st.button('Predict'):
#         text = text_cleaning(txt)
#         model = pickle.load(open('sentiment_model.pkl', 'rb'))
#         prediction = model.predict([text])
#         output = int(prediction[0])
#         probas = model.predict_proba([text])
#         output_probability = "{:.2f}".format(float(probas[:, output]))
#         sentiments = {0: "NEGATIVE", 1: "POSITIVE"}
#         st.write(f"It's {sentiments[output]} sentiment with {output_probability} probability")

if __name__ == "__main__":
    st.set_page_config(page_title="Sentiment Analysis")
    st.title("Sentiment Analysis")
    main()
