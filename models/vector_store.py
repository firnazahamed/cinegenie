import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import json
import numpy as np
from config.settings import CHROMA_PERSIST_DIRECTORY

class MovieVectorStore:
    def __init__(self, collection_name: str = "movie_recommendations"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIRECTORY,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        try:
            collection = self.client.get_collection(name=self.collection_name)
        except ValueError:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return collection
    
    def add_movies(self, 
                   movies: List[Dict], 
                   embeddings: List[List[float]], 
                   movie_texts: List[str]) -> None:
        if len(movies) != len(embeddings) or len(movies) != len(movie_texts):
            raise ValueError("Movies, embeddings, and texts must have the same length")
        
        ids = [str(movie.get('id', movie.get('movieId', idx))) 
               for idx, movie in enumerate(movies)]
        
        metadatas = []
        for movie in movies:
            metadata = self._prepare_metadata(movie)
            metadatas.append(metadata)
        
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=movie_texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(movies)} movies to vector store")
        except Exception as e:
            print(f"Error adding movies to vector store: {e}")
    
    def _prepare_metadata(self, movie: Dict) -> Dict[str, Any]:
        metadata = {}
        
        safe_fields = [
            'title', 'clean_title', 'release_date', 'year', 'runtime', 
            'vote_average', 'vote_count', 'popularity', 'adult',
            'avg_rating', 'rating_count', 'original_language'
        ]
        
        for field in safe_fields:
            if field in movie and movie[field] is not None:
                value = movie[field]
                if isinstance(value, (str, int, float, bool)):
                    if isinstance(value, float) and np.isnan(value):
                        continue
                    metadata[field] = value
        
        if 'genres' in movie:
            if isinstance(movie['genres'], list):
                if movie['genres'] and isinstance(movie['genres'][0], dict):
                    metadata['genres'] = '|'.join([g['name'] for g in movie['genres']])
                else:
                    metadata['genres'] = '|'.join(movie['genres'])
            elif isinstance(movie['genres'], str):
                metadata['genres'] = movie['genres']
        
        if 'genres_list' in movie and isinstance(movie['genres_list'], list):
            metadata['genres'] = '|'.join(movie['genres_list'])
        
        if 'overview' in movie and movie['overview']:
            metadata['has_overview'] = True
        
        if 'user_tags' in movie and movie['user_tags']:
            metadata['has_user_tags'] = True
        
        return metadata
    
    def search_similar_movies(self, 
                             query_embedding: List[float], 
                             n_results: int = 10,
                             where_filter: Optional[Dict] = None) -> Dict:
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,
                include=['metadatas', 'documents', 'distances']
            )
            return results
        except Exception as e:
            print(f"Error searching movies: {e}")
            return {"metadatas": [[]], "documents": [[]], "distances": [[]]}
    
    def get_movies_by_genre(self, genre: str, n_results: int = 10) -> Dict:
        where_filter = {
            "genres": {"$contains": genre}
        }
        return self.search_similar_movies(
            query_embedding=[0] * 1536,  # Dummy embedding for genre filtering
            n_results=n_results,
            where_filter=where_filter
        )
    
    def get_movies_by_rating_range(self, 
                                  min_rating: float = 0, 
                                  max_rating: float = 10,
                                  n_results: int = 10) -> Dict:
        where_filter = {
            "$and": [
                {"vote_average": {"$gte": min_rating}},
                {"vote_average": {"$lte": max_rating}}
            ]
        }
        return self.search_similar_movies(
            query_embedding=[0] * 1536,  # Dummy embedding for rating filtering
            n_results=n_results,
            where_filter=where_filter
        )
    
    def get_movies_by_runtime(self, 
                             max_runtime: int,
                             n_results: int = 10) -> Dict:
        where_filter = {
            "runtime": {"$lte": max_runtime}
        }
        return self.search_similar_movies(
            query_embedding=[0] * 1536,  # Dummy embedding for runtime filtering
            n_results=n_results,
            where_filter=where_filter
        )
    
    def get_collection_stats(self) -> Dict:
        try:
            count = self.collection.count()
            return {
                "total_movies": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"total_movies": 0, "collection_name": self.collection_name}
    
    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self._get_or_create_collection()
            print(f"Reset collection: {self.collection_name}")
        except Exception as e:
            print(f"Error resetting collection: {e}")
    
    def update_movie(self, movie_id: str, metadata: Dict, document: str = None) -> None:
        try:
            if document:
                self.collection.update(
                    ids=[movie_id],
                    metadatas=[metadata],
                    documents=[document]
                )
            else:
                self.collection.update(
                    ids=[movie_id],
                    metadatas=[metadata]
                )
            print(f"Updated movie {movie_id}")
        except Exception as e:
            print(f"Error updating movie {movie_id}: {e}")