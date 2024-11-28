# app.py
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer with caching"""
    try:
        model = load_model('LSTM_word.keras')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None

def predict_next_word(model, tokenizer, text, max_sequence_len):
    """Predict the next word given input text"""
    try:
        # Convert text to sequence
        token_list = tokenizer.texts_to_sequences([text])[0]
        
        # Handle sequence length
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]
            
        # Pad sequence
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Make prediction
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        
        # Get word from index
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
                
        return None
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def main():
    st.title("Next Word Prediction with LSTM")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Failed to load model or tokenizer. Please check if files exist.")
        return
    
    # Get max sequence length from model
    max_sequence_len = model.input_shape[1] + 1
    
    # Create input field with clear placeholder
    input_text = st.text_input(
        "Enter a sequence of words:",
        placeholder="Example: To be or not to",
        help="Enter a phrase and the model will predict the next word"
    )
    
    # Add prediction button
    if st.button("Predict Next Word", type="primary"):
        if not input_text:
            st.warning("Please enter some text first.")
            return
            
        with st.spinner("Predicting..."):
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            
        if next_word:
            st.success(f"Predicted next word: **{next_word}**")
            st.write(f"Complete phrase: **{input_text} {next_word}**")
        else:
            st.error("Could not predict the next word. Please try a different input.")
    
    # Add information about the model
    with st.expander("About this model"):
        st.write("""
        This is an LSTM-based next word prediction model trained on Shakespeare's Hamlet.
        It predicts the next word based on the sequence of words you provide.
        The model works best with phrases similar to those found in the original text.
        """)

if __name__ == "__main__":
    main()