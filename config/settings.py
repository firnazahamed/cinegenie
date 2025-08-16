import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

TMDB_ENDPOINTS = {
    "popular": "/movie/popular",
    "top_rated": "/movie/top_rated",
    "now_playing": "/movie/now_playing",
    "upcoming": "/movie/upcoming",
    "movie_details": "/movie/{movie_id}",
    "search": "/search/movie"
}

MOVIELENS_URLS = {
    "movies": "https://files.grouplens.org/datasets/movielens/ml-25m/movies.csv",
    "ratings": "https://files.grouplens.org/datasets/movielens/ml-25m/ratings.csv",
    "tags": "https://files.grouplens.org/datasets/movielens/ml-25m/tags.csv"
}