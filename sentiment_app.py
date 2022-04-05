from sentiment_model import text_cleaning
import pickle
import streamlit as st


def main():
    txt = st.text_area(label='Put your review here')

    if st.button('Predict'):
        text = text_cleaning(txt)
        model = pickle.load(open('sentiment_model.pkl', 'rb'))
        prediction = model.predict([text])
        output = int(prediction[0])
        probas = model.predict_proba([text])
        output_probability = "{:.2f}".format(float(probas[:, output]))
        sentiments = {0: "NEGATIVE", 1: "POSITIVE"}
        st.write(f"It's {sentiments[output]} sentiment with {output_probability} probability")

if __name__ == "__main__":
    st.set_page_config(page_title="Sentiment Analysis")
    st.title("Sentiment Analysis")
    main()
