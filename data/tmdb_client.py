import requests
import time
from typing import Dict, List, Optional
from config.settings import TMDB_API_KEY, TMDB_BASE_URL, TMDB_ENDPOINTS

class TMDBClient:
    def __init__(self):
        self.api_key = TMDB_API_KEY
        self.base_url = TMDB_BASE_URL
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        if not self.api_key:
            raise ValueError("TMDB API key not configured")
            
        default_params = {"api_key": self.api_key}
        if params:
            default_params.update(params)
            
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=default_params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to TMDB: {e}")
            return None
    
    def get_popular_movies(self, page: int = 1) -> List[Dict]:
        endpoint = TMDB_ENDPOINTS["popular"]
        result = self._make_request(endpoint, {"page": page})
        return result.get("results", []) if result else []
    
    def get_top_rated_movies(self, page: int = 1) -> List[Dict]:
        endpoint = TMDB_ENDPOINTS["top_rated"]
        result = self._make_request(endpoint, {"page": page})
        return result.get("results", []) if result else []
    
    def get_now_playing_movies(self, page: int = 1) -> List[Dict]:
        endpoint = TMDB_ENDPOINTS["now_playing"]
        result = self._make_request(endpoint, {"page": page})
        return result.get("results", []) if result else []
    
    def get_upcoming_movies(self, page: int = 1) -> List[Dict]:
        endpoint = TMDB_ENDPOINTS["upcoming"]
        result = self._make_request(endpoint, {"page": page})
        return result.get("results", []) if result else []
    
    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        endpoint = TMDB_ENDPOINTS["movie_details"].format(movie_id=movie_id)
        params = {"append_to_response": "credits,keywords,reviews"}
        return self._make_request(endpoint, params)
    
    def search_movies(self, query: str, page: int = 1) -> List[Dict]:
        endpoint = TMDB_ENDPOINTS["search"]
        params = {"query": query, "page": page}
        result = self._make_request(endpoint, params)
        return result.get("results", []) if result else []
    
    def get_multiple_pages(self, fetch_function, max_pages: int = 5, delay: float = 0.5) -> List[Dict]:
        all_movies = []
        for page in range(1, max_pages + 1):
            movies = fetch_function(page=page)
            if not movies:
                break
            all_movies.extend(movies)
            if page < max_pages:
                time.sleep(delay)
        return all_movies
    
    def fetch_comprehensive_movie_data(self, max_pages_per_category: int = 3) -> List[Dict]:
        all_movies = []
        
        categories = [
            self.get_popular_movies,
            self.get_top_rated_movies,
            self.get_now_playing_movies,
            self.get_upcoming_movies
        ]
        
        for category_func in categories:
            movies = self.get_multiple_pages(category_func, max_pages_per_category)
            all_movies.extend(movies)
            time.sleep(1)
        
        unique_movies = {movie["id"]: movie for movie in all_movies}
        return list(unique_movies.values())
    
    def get_detailed_movie_info(self, movie_ids: List[int], delay: float = 0.5) -> List[Dict]:
        detailed_movies = []
        for movie_id in movie_ids:
            details = self.get_movie_details(movie_id)
            if details:
                detailed_movies.append(details)
            time.sleep(delay)
        return detailed_movies