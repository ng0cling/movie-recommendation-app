import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import gdown

st.set_page_config(page_title="🎬 Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("🔹 Recommend movies similar to your favorite one!")

# === Step 1: Download movie_data.pkl from Google Drive ===
file_path = "movie_data.pkl"
gdrive_url = "https://drive.google.com/uc?id=1_X2aSXG5zNPM1MMDoNUku7UyXA0Bhghk"

if not os.path.exists(file_path):
    with st.spinner("📥 Downloading data from Google Drive (~500MB)... Please wait."):
        gdown.download(gdrive_url, file_path, quiet=False)
    st.success("✅ Download completed successfully!")

# === Step 2: Load pickle data ===
with open(file_path, "rb") as file:
    movies, cosine_sim = pickle.load(file)

# === Step 3: Define functions ===
def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = movies[movies["title"] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Top 10
        movie_indices = [i[0] for i in sim_scores]
        return movies[["title", "movie_id"]].iloc[movie_indices]
    except:
        return pd.DataFrame(columns=["title", "movie_id"])

def fetch_poster(movie_id):
    api_key = "7b995d3c6fd91a2284b4ad8cb390c7b8"  # Replace with your TMDB API key
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/300x450?text=No+Image"

# === Step 4: Streamlit UI ===
selected_movie = st.selectbox("🎞️ Select a movie:", movies["title"].values)

if st.button("Get Recommendations 🚀"):
    with st.spinner("Generating recommendations..."):
        recommendations = get_recommendations(selected_movie)

    if recommendations.empty:
        st.error("❌ Movie not found in dataset.")
    else:
        st.subheader(f"🎯 Top 10 movies similar to **{selected_movie}**:")

        # Display 2 rows × 5 columns (10 movies)
        for i in range(0, 10, 5):
            cols = st.columns(5)
            for col, j in zip(cols, range(i, i+5)):
                if j < len(recommendations):
                    movie_title = recommendations.iloc[j]["title"]
                    movie_id = recommendations.iloc[j]["movie_id"]
                    poster_url = fetch_poster(movie_id)
                    with col:
                        st.image(poster_url, width=140)
                        st.caption(movie_title)

