import streamlit as st
import pandas as pd
import numpy as np
import joblib


pipe_lr=joblib.load(open("emotion_classifier_pipe_lr_16_dec_2024.pkl","rb"))

def predict_emotions(docx):
    results=pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results_proba=pipe_lr.predict_proba([docx])
    return results_proba


def main():
    st.title("Emotion Text Analyser")

    with st.form(key='emotion form'):
        raw_text=st.text_area("Enter Text Here")
        submit_text=st.form_submit_button(label='Submit')
    
    if submit_text:
        col1,col2=st.columns(2)
        
        #applying functions
        prediction=predict_emotions(raw_text)
        probability=get_prediction_proba(raw_text)
        with col1:
            st.success("Original Text")
            st.write(raw_text)
            st.success("Prediction")
            st.write(prediction)


        with col2:
            st.success("Prediction Probability")
            
            proba_df=pd.DataFrame(probability,columns=pipe_lr.classes_)
            st.write(proba_df)

if __name__=='__main__':
    main()