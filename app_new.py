import streamlit as st
import pandas as pd
import numpy as np
import ast
import pickle
import os
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# 🎬 1. Xử lý và cache dữ liệu
# ===============================
@st.cache_data
def load_and_process_data():
    if os.path.exists('movie_data.pkl'):
        with open('movie_data.pkl', 'rb') as f:
            movies, cosine_sim = pickle.load(f)
        return movies, cosine_sim

    st.info("🔄 Processing movie data for the first time... This may take 1–2 minutes.")

    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = pd.read_csv('tmdb_5000_movies.csv')

    # --- Xử lý dữ liệu ---
    movies = movies.merge(credits, left_on='title', right_on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]])
    movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])
    movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
    movies = movies[['movie_id', 'title', 'overview', 'tags']]
    movies['tags'] = movies['tags'].apply(lambda x: x.lower())

    # --- TF-IDF + Cosine Similarity ---
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # --- Lưu file ---
    with open('movie_data.pkl', 'wb') as f:
        pickle.dump((movies, cosine_sim), f)

    st.success("✅ Data processing completed!")
    return movies, cosine_sim


# ===============================
# 🎯 2. Hàm gợi ý phim
# ===============================
def get_recommendations(title, cosine_sim, movies):
    try:
        idx = movies[movies['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movies[['title', 'movie_id']].iloc[movie_indices]
    except:
        return pd.DataFrame(columns=['title', 'movie_id'])


# ===============================
# 🖼️ 3. Hàm lấy poster từ TMDB API
# ===============================
def fetch_poster(movie_id):
    api_key = '7b995d3c6fd91a2284b4ad8cb390c7b8'  # 🔑 thay bằng API key của bạn
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}'
    response = requests.get(url)
    data = response.json()
    poster_path = data.get('poster_path')
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    else:
        return "https://via.placeholder.com/500x750?text=No+Image"


# ===============================
# 🧠 4. Giao diện Streamlit
# ===============================
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation System")
st.markdown("---")

# Load or process data
movies, cosine_sim = load_and_process_data()

# Movie selection
selected_movie = st.selectbox("🎭 Select a movie to get recommendations:", movies['title'].values)

if st.button("🚀 Get Recommendations"):
    recommendations = get_recommendations(selected_movie, cosine_sim, movies)

    st.subheader(f"🎯 Recommendations for: **{selected_movie}**")

    # Display posters in 2x5 grid
    for i in range(0, len(recommendations), 5):
        cols = st.columns(5)
        for col, (_, row) in zip(cols, recommendations.iloc[i:i+5].iterrows()):
            with col:
                poster_url = fetch_poster(row['movie_id'])
                st.image(poster_url, width=150)
                st.caption(row['title'])

# Data overview
st.markdown("---")
st.subheader("📊 Movie Data Overview")
col1, col2 = st.columns(2)

with col1:
    st.write("**First 5 movies:**")
    st.dataframe(movies[['title', 'tags']].head())

with col2:
    st.write("**Dataset info:**")
    st.write(f"Total movies: {len(movies)}")
    st.write(f"Columns: {list(movies.columns)}")
