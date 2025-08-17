import pandas as pd
from typing import List, Dict, Optional, Tuple
import re
from data.tmdb_client import TMDBClient
from data.movielens_client import MovieLensClient
from models.embeddings import OpenAIClient
from models.vector_store import MovieVectorStore

class MovieDataProcessor:
    def __init__(self):
        self.tmdb_client = TMDBClient()
        self.movielens_client = MovieLensClient()
        self.openai_client = OpenAIClient()
        self.vector_store = MovieVectorStore()
    
    def create_movie_description(self, movie: Dict) -> str:
        description_parts = []
        
        title = movie.get('title', movie.get('clean_title', 'Unknown'))
        description_parts.append(f"Title: {title}")
        
        year = movie.get('year', movie.get('release_date', ''))
        if year:
            description_parts.append(f"Year: {year}")
        
        genres = movie.get('genres', movie.get('genres_list', []))
        if genres:
            if isinstance(genres, list):
                if genres and isinstance(genres[0], dict):
                    genre_names = [g['name'] for g in genres]
                else:
                    genre_names = genres
                description_parts.append(f"Genres: {', '.join(genre_names)}")
            elif isinstance(genres, str) and genres != "(no genres listed)":
                description_parts.append(f"Genres: {genres.replace('|', ', ')}")
        
        runtime = movie.get('runtime')
        if runtime and runtime > 0:
            description_parts.append(f"Runtime: {runtime} minutes")
        
        rating = movie.get('vote_average', movie.get('avg_rating'))
        if rating and rating > 0:
            description_parts.append(f"Rating: {rating}/10")
        
        overview = movie.get('overview', '')
        if overview:
            description_parts.append(f"Plot: {overview}")
        
        user_tags = movie.get('user_tags', '')
        if user_tags:
            description_parts.append(f"User tags: {user_tags}")
        
        if 'production_companies' in movie and movie['production_companies']:
            companies = [comp['name'] for comp in movie['production_companies'][:3]]
            description_parts.append(f"Production: {', '.join(companies)}")
        
        if 'spoken_languages' in movie and movie['spoken_languages']:
            languages = [lang['english_name'] for lang in movie['spoken_languages'][:2]]
            description_parts.append(f"Languages: {', '.join(languages)}")
        
        if 'keywords' in movie and movie['keywords']:
            if 'keywords' in movie['keywords']:
                keywords = [kw['name'] for kw in movie['keywords']['keywords'][:5]]
                description_parts.append(f"Keywords: {', '.join(keywords)}")
        
        return ' | '.join(description_parts)
    
    def merge_tmdb_movielens_data(self, tmdb_movies: List[Dict], movielens_df: pd.DataFrame) -> List[Dict]:
        merged_movies = []
        
        for tmdb_movie in tmdb_movies:
            tmdb_title = tmdb_movie.get('title', '').lower()
            tmdb_year = None
            
            if 'release_date' in tmdb_movie and tmdb_movie['release_date']:
                try:
                    tmdb_year = int(tmdb_movie['release_date'][:4])
                except (ValueError, TypeError):
                    pass
            
            best_match = None
            best_score = 0
            
            for _, ml_row in movielens_df.iterrows():
                ml_title = str(ml_row.get('clean_title', '')).lower()
                ml_year = ml_row.get('year')
                
                if ml_year:
                    try:
                        ml_year = int(ml_year)
                    except (ValueError, TypeError):
                        ml_year = None
                
                title_similarity = self._calculate_title_similarity(tmdb_title, ml_title)
                year_match = (tmdb_year and ml_year and abs(tmdb_year - ml_year) <= 1)
                
                score = title_similarity
                if year_match:
                    score += 0.3
                
                if score > best_score and score > 0.7:
                    best_score = score
                    best_match = ml_row
            
            if best_match is not None:
                merged_movie = tmdb_movie.copy()
                merged_movie.update({
                    'movielens_id': best_match.get('movieId'),
                    'avg_rating': best_match.get('avg_rating', 0),
                    'rating_count': best_match.get('rating_count', 0),
                    'user_tags': best_match.get('user_tags', ''),
                    'movielens_genres': best_match.get('genres', '')
                })
                merged_movies.append(merged_movie)
            else:
                merged_movies.append(tmdb_movie)
        
        return merged_movies
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        title1_clean = re.sub(r'[^\w\s]', '', title1).strip()
        title2_clean = re.sub(r'[^\w\s]', '', title2).strip()
        
        if title1_clean == title2_clean:
            return 1.0
        
        words1 = set(title1_clean.split())
        words2 = set(title2_clean.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def process_and_store_movies(self, max_movies: int = 1000) -> Dict[str, int]:
        print("Fetching TMDB movies...")
        tmdb_movies = self.tmdb_client.fetch_comprehensive_movie_data(max_pages_per_category=3)
        
        detailed_movies = []
        for i, movie in enumerate(tmdb_movies[:max_movies]):
            if i % 50 == 0:
                print(f"Fetching detailed info for movie {i+1}/{len(tmdb_movies[:max_movies])}")
            
            detailed_movie = self.tmdb_client.get_movie_details(movie['id'])
            if detailed_movie:
                detailed_movies.append(detailed_movie)
        
        print("Loading MovieLens data...")
        movielens_df = self.movielens_client.filter_popular_movies(min_ratings=10)
        
        print("Merging TMDB and MovieLens data...")
        merged_movies = self.merge_tmdb_movielens_data(detailed_movies, movielens_df)
        
        print("Creating movie descriptions...")
        movie_texts = []
        for movie in merged_movies:
            description = self.create_movie_description(movie)
            movie_texts.append(description)
        
        print("Generating embeddings...")
        embeddings = self.openai_client.get_embeddings_batch(movie_texts, batch_size=50)
        
        print("Storing in vector database...")
        self.vector_store.add_movies(merged_movies, embeddings, movie_texts)
        
        stats = self.vector_store.get_collection_stats()
        
        return {
            "tmdb_movies": len(tmdb_movies),
            "detailed_movies": len(detailed_movies),
            "merged_movies": len(merged_movies),
            "stored_movies": stats["total_movies"]
        }
    
    def get_processing_status(self) -> Dict:
        stats = self.vector_store.get_collection_stats()
        return {
            "vector_store_movies": stats["total_movies"],
            "collection_name": stats["collection_name"]
        }
    
    def reset_and_reprocess(self, max_movies: int = 1000) -> Dict[str, int]:
        print("Resetting vector store...")
        self.vector_store.reset_collection()
        
        return self.process_and_store_movies(max_movies)