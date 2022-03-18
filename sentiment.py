from cProfile import label
import streamlit as st

def main():
    txt = st.text_area(label='Put your review here')
    st.write('Sentiment:',txt)
        

if __name__ == "__main__":
    st.set_page_config(page_title="Sentiment Analysis")
    st.title("Sentiment Analysis")
    main()
