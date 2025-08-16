import pandas as pd
import requests
import os
from typing import Dict, List, Optional
from config.settings import MOVIELENS_URLS

class MovieLensClient:
    def __init__(self, data_dir: str = "./data/movielens"):
        self.data_dir = data_dir
        self.urls = MOVIELENS_URLS
        os.makedirs(data_dir, exist_ok=True)
        
    def download_file(self, url: str, filename: str) -> str:
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"File {filename} already exists, skipping download")
            return filepath
            
        print(f"Downloading {filename}...")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded {filename} successfully")
        return filepath
    
    def download_movielens_data(self) -> Dict[str, str]:
        filepaths = {}
        
        for data_type, url in self.urls.items():
            filename = f"{data_type}.csv"
            try:
                filepath = self.download_file(url, filename)
                filepaths[data_type] = filepath
            except Exception as e:
                print(f"Error downloading {data_type}: {e}")
                
        return filepaths
    
    def load_movies(self) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, "movies.csv")
        if not os.path.exists(filepath):
            self.download_movielens_data()
            
        movies_df = pd.read_csv(filepath)
        
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')[0]
        movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
        movies_df['genres_list'] = movies_df['genres'].str.split('|')
        
        return movies_df
    
    def load_ratings(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, "ratings.csv")
        if not os.path.exists(filepath):
            self.download_movielens_data()
            
        if sample_size:
            ratings_df = pd.read_csv(filepath, nrows=sample_size)
        else:
            ratings_df = pd.read_csv(filepath)
            
        return ratings_df
    
    def load_tags(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, "tags.csv")
        if not os.path.exists(filepath):
            self.download_movielens_data()
            
        if sample_size:
            tags_df = pd.read_csv(filepath, nrows=sample_size)
        else:
            tags_df = pd.read_csv(filepath)
            
        return tags_df
    
    def get_movie_stats(self) -> pd.DataFrame:
        movies_df = self.load_movies()
        ratings_df = self.load_ratings()
        
        rating_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count', 'std']
        }).round(2)
        
        rating_stats.columns = ['avg_rating', 'rating_count', 'rating_std']
        rating_stats = rating_stats.reset_index()
        
        movie_stats = movies_df.merge(rating_stats, on='movieId', how='left')
        movie_stats['avg_rating'] = movie_stats['avg_rating'].fillna(0)
        movie_stats['rating_count'] = movie_stats['rating_count'].fillna(0)
        movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)
        
        return movie_stats
    
    def get_movie_tags_summary(self) -> pd.DataFrame:
        tags_df = self.load_tags()
        
        tag_summary = tags_df.groupby('movieId')['tag'].apply(
            lambda x: ' | '.join(x.unique())
        ).reset_index()
        
        tag_summary.columns = ['movieId', 'user_tags']
        return tag_summary
    
    def get_comprehensive_movie_data(self) -> pd.DataFrame:
        movie_stats = self.get_movie_stats()
        tag_summary = self.get_movie_tags_summary()
        
        comprehensive_data = movie_stats.merge(
            tag_summary, on='movieId', how='left'
        )
        
        comprehensive_data['user_tags'] = comprehensive_data['user_tags'].fillna('')
        
        return comprehensive_data
    
    def filter_popular_movies(self, min_ratings: int = 50) -> pd.DataFrame:
        comprehensive_data = self.get_comprehensive_movie_data()
        popular_movies = comprehensive_data[
            comprehensive_data['rating_count'] >= min_ratings
        ].copy()
        
        return popular_movies.sort_values('avg_rating', ascending=False)