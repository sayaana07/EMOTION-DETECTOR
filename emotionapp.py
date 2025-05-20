import os
os.environ["STREAMLIT_WATCHER"] = "false"

import streamlit as st
from transformers import pipeline

# Load the emotion detection model once
@st.cache_resource
def load_model():
    return pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

emotion_model = load_model()

# App title
st.title("🧠 AI Emotion Detector")
st.markdown("Type anything below and let the AI guess how you're feeling!")

# Input text box
user_input = st.text_area("Enter your text here 👇", "")

# Detect button
if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please type something first!")
    else:
        # Run model
        results = emotion_model(user_input)[0]
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

        # Top emotion
        top = sorted_results[0]
        st.markdown(f"### 😎 Detected Emotion: **{top['label'].capitalize()}**")
        st.progress(top['score'])

        # Show all emotions
        st.subheader("Full Emotion Breakdown:")
        for res in sorted_results:
            st.write(f"**{res['label'].capitalize()}**: {res['score']:.2f}")
