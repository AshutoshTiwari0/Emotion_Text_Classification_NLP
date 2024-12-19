# Emotion Classifier Project

This project involves building a machine learning model to classify text into various emotional categories. The emotions covered include joy, sadness, fear, anger, surprise, disgust, shame, and neutral. The model is implemented using Python and trained on an emotion-labeled dataset.

---

## Table of Contents
1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)  
4. [Model Training](#model-training)  
5. [Evaluation](#evaluation)  
6. [Usage](#usage)  
7. [Streamlit App](#streamlit-app)  
8. [File Structure](#file-structure)  
9. [Dependencies](#dependencies)  

---

## Overview
This project utilizes natural language processing (NLP) techniques to classify text into one of eight emotional categories. The model employs a logistic regression pipeline and is designed to make predictions based on cleaned and preprocessed text data.

---

## Dataset
- **Source**: `emotion_dataset_2.csv`  
- **Columns**:  
  - `Emotion`: Target label (e.g., joy, sadness, anger)  
  - `Text`: Raw input text  

---

## Data Cleaning and Preprocessing
1. **User Handles Removal**: Text containing mentions (e.g., `@username`) was cleaned.  
2. **Non-Alphabetical Characters Removal**: Numbers, symbols, and special characters were removed.  
3. **Stopwords Removal**: Common English words (e.g., "is", "the") were removed to enhance feature extraction.  
4. **Splitting**: Data was split into training and testing sets with a 70:30 ratio.

---

## Model Training
- **Algorithm**: Logistic Regression  
- **Pipeline**: Combines `CountVectorizer` for text vectorization and `LogisticRegression` for classification.  
- **Training**: The model was trained on the cleaned dataset.  

---

## Evaluation
- **Accuracy**: ~63.03% on the test set.  
- **Example Predictions**:  
  - *Input*: `"This book was so interesting that it made me happy"`  
    *Prediction*: `joy`  
  - *Input*: `"Why you don’t do your work correctly?"`  
    *Prediction*: `anger`  

---

## Usage
1. **Prediction**:  
   Use the model to predict the emotion of any text input:
   ```python
   pipe_lr.predict(["Your input text here"])Save Model:
The trained pipeline is saved as a .pkl file:


## Streamlit App:
A Streamlit-based web application has been developed for user-friendly interaction with the model. The app allows users to input text and get emotion predictions in real time.

##File Structure
emotion_dataset_2.csv: Dataset used for training and testing.
emotion_classifier_pipe_lr_16_dec_2024.pkl: Trained model pipeline.
app.py: Streamlit app code.

## Dependencies
# Libraries:
1.numpy<br>
2.pandas<br>
3.seaborn<br>
4.matplotlib<br>
5.sklearn<br>
6.nltk<br>
7.joblib<br>
8.streamlit<br>

Install all required dependencies:

```bash
pip install -r requirements.txt
```
Live Link-> https://emotion-text-classification-nlp.streamlit.app/
Web App Preview
![image](https://github.com/user-attachments/assets/e71c127a-79e1-4f0e-9d1c-67919e30cbaf)



