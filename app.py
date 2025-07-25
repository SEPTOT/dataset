import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, label_encoders
with open('music_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Ambil label encoder target
le_target = label_encoders['Genre']

st.title("🎵 Genre Music Prediction App")

st.write("Upload informasi musik favoritmu dan dapatkan prediksi genre!")

# Input user
song_title = st.text_input("Song Title", "Shape of You")
artist = st.text_input("Artist", "Ed Sheeran")
release_year = st.number_input("Release Year", min_value=1900, max_value=2025, value=2017)
duration = st.number_input("Duration (Minutes)", min_value=0.0, max_value=15.0, value=4.0)
listened_year = st.number_input("Listened Year", min_value=2000, max_value=2025, value=2023)
listened_month = st.number_input("Listened Month", min_value=1, max_value=12, value=7)
listened_day = st.number_input("Listened Day", min_value=1, max_value=31, value=15)
platform = st.text_input("Platform", "Spotify")

if st.button("Predict Genre"):
    # Encode input
    song_title_enc = label_encoders['Song_Title'].transform([song_title])[0] if song_title in label_encoders['Song_Title'].classes_ else 0
    artist_enc = label_encoders['Artist'].transform([artist])[0] if artist in label_encoders['Artist'].classes_ else 0
    platform_enc = label_encoders['Platform'].transform([platform])[0] if platform in label_encoders['Platform'].classes_ else 0

    # Susun fitur sesuai training
    data = np.array([[song_title_enc, artist_enc, release_year, duration, listened_year, listened_month, listened_day, platform_enc]])

    # Scaling
    data_scaled = scaler.transform(data)

    # Predict
    prediction = model.predict(data_scaled)
    predicted_genre = le_target.inverse_transform(prediction)[0]

    st.success(f"The predicted genre for this song is: **{predicted_genre}** 🎶")
