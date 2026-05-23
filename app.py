import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('emotion_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Page settings
st.set_page_config(
    page_title="Emotion Analyzer",
    page_icon="😊",
    layout="centered"
)
st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: white;
}

h1 {
    color: #00FFD1;
    text-align: center;
}

textarea {
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("😊 Emotion Analysis App")

st.write("Analyze emotions from text using NLP and Machine Learning.")

# Input
user_input = st.text_area("Enter Your Text")

# Emotion emojis
emotion_map = {
    0: 'sadness 😢',
    1: 'anger 😡',
    2: 'love ❤️',
    3: 'surprise 😲',
    4: 'fear 😨',
    5: 'joy 😂'
}

# Prediction
if st.button("Analyze Emotion"):

    if user_input.strip() != "":

        transformed_text = vectorizer.transform([user_input])

        prediction_num = model.predict(transformed_text)[0]
        prediction = emotion_map[prediction_num]

        # Confidence score
        prob = model.predict_proba(transformed_text)

        confidence = prob.max() * 100

        st.success(f"Predicted Emotion: {prediction}")

        st.write(f"Confidence Score: {confidence:.2f}%")

    else:
        st.warning("Please enter some text.")

#Sidebar
st.sidebar.title("About:")

st.sidebar.info(
    """ An end-to-end NLP application that leverages 
    Machine Learning algorithms for intelligent emotion classification 
    from textual data.
    
    ⚙️ Tech Stack:
    • Python  
    • Scikit-learn  
    • NLP Vectorization  
    • Logistic Regression  
    • Streamlit Cloud  
    
    🔍 Features:
    ✔ Real-time emotion prediction  
    ✔ Confidence score analysis  
    ✔ Interactive UI  
    ✔ Lightweight ML inference pipeline  
    
    Designed for AI-driven sentiment and behavioral analysis applications.
    """
)
#Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
