import pickle
import streamlit as st
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

# Load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizers = pickle.load(handle)

models = {
    'LASTM 1 LAYER MODEL': load_model('lastm-1-layer-best_model.h5'),
    'LASTM 2 LAYER MODEL': load_model('lastm-2-layer-best_model.h5'),
    'GRU MODEL': load_model('gru-best_model.h5'),
    'CNN + LSTM MODEL': load_model('cnn+lastm-best_model.h5'),
}

st.title("Suicide Detection Web App")


st.markdown("<h2 style='color: black; font-style: normal ;text-decoration: underline'>Enter text</h2>", unsafe_allow_html=True)
text = st.text_area("")
if st.button("Predict"):
    if text:
        predictions = {}
        max_percentage = 0
        most_likely_prediction = ''

        for model_name, model in models.items():
            twt = [text]
            twt = tokenizers.texts_to_sequences(twt)
            twt = pad_sequences(twt, maxlen=60, dtype='int32')

            predicted = model.predict(twt, batch_size=1, verbose=True)
            percentage = predicted[0][0]  # Percentage of Potential Suicide Post

            if percentage > max_percentage:
                max_percentage = percentage
                most_likely_prediction = "Suicide" if percentage > 0.5 else "Non-Suicide"

    
        st.markdown("<h2 style='font-style:normal;color:black;text-decoration: underline'>Pie Chart</h3>", unsafe_allow_html=True)

        # Create and display a pie chart
        fig, ax = plt.subplots()
        labels = ['Potential Suicide', 'Non-Suicide']
        sizes = [max_percentage, 1 - max_percentage]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        
        st.markdown("<h2 style='font-style: Normal ;color :black ;text-decoration: underline'>Prediction</h2>", unsafe_allow_html=True)

        # Display the most likely prediction in large font
        st.markdown(f"<h1 style='font-size: 24px; color: black;'>Most Likely Prediction: {most_likely_prediction}</h1>", unsafe_allow_html=True)
