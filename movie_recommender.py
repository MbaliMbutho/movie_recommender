import streamlit as st
import requests
import cohere
import numpy as np
import json
import os
import random

# ==============================
# CONFIG
# ==============================
COHERE_API_KEY = "NqCDyPmfZHiXEDiyn0Xooutz67b0XHFPoeZ8qeYy"   # 🔑 Replace with your Cohere API key
OMDB_API_KEY = "740a1849"       # 🔑 Replace with your OMDb API key
CACHE_FILE = "embeddings_cache.json"

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

# ==============================
# CACHE FUNCTIONS
# ==============================
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

emb_cache = load_cache()

def get_embedding_cached(text, input_type="search_document"):
    """Get embedding from cache or Cohere API"""
    if text in emb_cache:
        return np.array(emb_cache[text])
    response = co.embed(
        texts=[text],
        model="embed-multilingual-v3.0",
        input_type=input_type
    )
    emb = np.array(response.embeddings[0])
    emb_cache[text] = emb.tolist()  # save as list for JSON
    save_cache(emb_cache)
    return emb

# ==============================
# OMDb FUNCTIONS
# ==============================
def fetch_movie_details(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}&plot=short"
    response = requests.get(url).json()
    if response.get("Response") == "True":
        return {
            "title": response.get("Title", "Unknown"),
            "year": response.get("Year", "N/A"),
            "genre": response.get("Genre", "N/A"),
            "plot": response.get("Plot", "N/A"),
            "poster": response.get("Poster", ""),
            "rating": response.get("imdbRating", "N/A"),
            "actors": response.get("Actors", "")
        }
    return None

def search_movies(query):
    """Search movies dynamically from OMDb with multiple pages and randomness"""
    titles = []
    for page in range(1, 4):  # fetch up to 30 results
        url = f"http://www.omdbapi.com/?s={query}&apikey={OMDB_API_KEY}&page={page}"
        response = requests.get(url).json()
        if response.get("Response") == "True":
            for item in response.get("Search", []):
                titles.append(item["Title"])
    return list(set(titles))

# ==============================
# RECOMMENDATION FUNCTION
# ==============================
def recommend_movies(user_inputs):
    query = user_inputs["genre"] or user_inputs["actor"] or user_inputs["mood"] or "movie"
    movie_titles = search_movies(query)
    if not movie_titles:
        return []

    random.shuffle(movie_titles)
    movie_titles = movie_titles[:20]

    # Mood → multiple genres
    mood_to_genre = {
        "happy": ["Comedy", "Family"],
        "adventurous": ["Action", "Adventure"],
        "relaxed": ["Romance", "Drama"],
        "scared": ["Horror", "Thriller"],
        "thoughtful": ["Drama", "Mystery"]
    }
    if not user_inputs["genre"] and user_inputs["mood"].lower() in mood_to_genre:
        user_inputs["genre"] = random.choice(mood_to_genre[user_inputs["mood"].lower()])

    # Fetch movie details
    movies = []
    combined_texts = []
    for title in movie_titles:
        movie = fetch_movie_details(title)
        if movie:
            movies.append(movie)
            combined_text = f"{movie['title']} ({movie['year']}). Genre: {movie['genre']}. Actors: {movie['actors']}. Plot: {movie['plot']}. Mood: {user_inputs['mood']}"
            combined_texts.append(combined_text)

    if not movies:
        return []

    # Batch embeddings for all movies
    # First add user query embedding
    query_text = f"Genre: {user_inputs['genre']}. Actor: {user_inputs['actor']}. Mood: {user_inputs['mood']}"
    all_texts = [query_text] + combined_texts

    # Check cache for batch embeddings
    embeddings = []
    for text in all_texts:
        embeddings.append(get_embedding_cached(text))

    user_emb = embeddings[0]
    movie_embs = embeddings[1:]

    # Score movies
    scored_movies = []
    for movie, movie_emb in zip(movies, movie_embs):
        score = 0
        if user_inputs['genre'] and user_inputs['genre'].lower() in movie['genre'].lower():
            score += 1
        if user_inputs['actor'] and user_inputs['actor'].lower() in movie['actors'].lower():
            score += 1
        if user_inputs['mood'] and user_inputs['mood'].lower() in movie['plot'].lower():
            score += 1

        similarity = np.dot(user_emb, movie_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(movie_emb))
        total_score = similarity + score
        scored_movies.append((total_score, movie))

    scored_movies.sort(reverse=True, key=lambda x: x[0])
    return [m for _, m in scored_movies[:3]]

# ==============================
# STREAMLIT APP
# ==============================
st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")
st.title("🤖 AI Movie Recommendation Assistant")
st.write("Chat with me to discover movies you'll love! 🍿")

if "favorites" not in st.session_state:
    st.session_state["favorites"] = []

with st.chat_message("assistant"):
    st.write("👋 Hey there! Let’s find you a great movie.")
    st.write("Tell me one of these: a favorite genre, actor, or your mood!")

genre = st.text_input("🎭 Genre you like (optional):")
actor = st.text_input("🎬 Favorite Actor (optional):")
mood = st.text_input("😊 Your Mood (e.g., happy, adventurous, relaxed):")

if st.button("✨ Get Recommendations"):
    user_inputs = {"genre": genre, "actor": actor, "mood": mood}
    recommendations = recommend_movies(user_inputs)

    if recommendations:
        st.subheader("🎯 Here are my top picks for you:")
        for rec in recommendations:
            cols = st.columns([1, 2])
            with cols[0]:
                if rec["poster"] and rec["poster"] != "N/A":
                    st.image(rec["poster"], width=150)
            with cols[1]:
                st.write(f"### {rec['title']} ({rec['year']})")
                st.write(f"**Genre:** {rec['genre']}")
                st.write(f"**IMDB Rating:** ⭐ {rec['rating']}")
                st.write(f"**Plot:** {rec['plot']}")
                trailer_url = f"https://www.youtube.com/results?search_query={rec['title']}+trailer"
                st.markdown(f"[▶️ Watch Trailer]({trailer_url})")

                if st.button(f"❤️ Add {rec['title']} to Favorites"):
                    if rec["title"] not in st.session_state["favorites"]:
                        st.session_state["favorites"].append(rec["title"])
            st.markdown("---")

        if st.session_state["favorites"]:
            st.subheader("❤️ Your Favorites")
            st.write(", ".join(st.session_state["favorites"]))
    else:
        st.warning("😅 No matches found. Try another genre, actor, or mood!")
