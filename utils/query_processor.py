import re
from typing import Dict, List, Optional, Tuple
from models.embeddings import OpenAIClient
from models.vector_store import MovieVectorStore

class QueryProcessor:
    def __init__(self):
        self.openai_client = OpenAIClient()
        self.vector_store = MovieVectorStore()
        
        self.genre_keywords = {
            'horror': ['horror', 'scary', 'frightening', 'terrifying', 'creepy', 'spooky'],
            'comedy': ['comedy', 'funny', 'hilarious', 'humor', 'laugh'],
            'action': ['action', 'adventure', 'fighting', 'chase', 'explosive'],
            'drama': ['drama', 'dramatic', 'emotional', 'serious'],
            'romance': ['romance', 'romantic', 'love', 'relationship'],
            'thriller': ['thriller', 'suspense', 'tension', 'mystery'],
            'sci-fi': ['sci-fi', 'science fiction', 'futuristic', 'space', 'alien'],
            'fantasy': ['fantasy', 'magical', 'magic', 'supernatural'],
            'crime': ['crime', 'criminal', 'detective', 'police', 'investigation'],
            'war': ['war', 'military', 'battle', 'soldier'],
            'western': ['western', 'cowboy', 'frontier'],
            'documentary': ['documentary', 'real life', 'true story', 'factual'],
            'animation': ['animated', 'animation', 'cartoon'],
            'musical': ['musical', 'music', 'singing', 'songs']
        }
        
        self.real_events_keywords = [
            'true story', 'based on true events', 'real events', 'real life',
            'biographical', 'biopic', 'historical', 'actual events',
            'true events', 'real story', 'factual', 'documentary'
        ]
        
        self.rating_keywords = {
            'high': ['highly rated', 'top rated', 'best', 'excellent', 'outstanding'],
            'good': ['good', 'well rated', 'recommended'],
            'recent': ['recent', 'new', 'latest', 'current', 'modern']
        }
    
    def extract_query_features(self, query: str) -> Dict:
        query_lower = query.lower()
        features = {
            'genres': [],
            'runtime_max': None,
            'rating_min': None,
            'real_events': False,
            'year_min': None,
            'year_max': None,
            'quality_preference': None,
            'original_query': query
        }
        
        for genre, keywords in self.genre_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                features['genres'].append(genre)
        
        runtime_match = re.search(r'under (\d+) (?:hours?|minutes?)', query_lower)
        if runtime_match:
            time_value = int(runtime_match.group(1))
            if 'hour' in runtime_match.group(0):
                features['runtime_max'] = time_value * 60
            else:
                features['runtime_max'] = time_value
        
        runtime_match_alt = re.search(r'(?:less than|under|below) (\d+)\s*(?:h|hr|hours?|min|minutes?)', query_lower)
        if runtime_match_alt:
            time_value = int(runtime_match_alt.group(1))
            unit = runtime_match_alt.group(0)
            if any(h in unit for h in ['h', 'hr', 'hour']):
                features['runtime_max'] = time_value * 60
            else:
                features['runtime_max'] = time_value
        
        if any(keyword in query_lower for keyword in self.real_events_keywords):
            features['real_events'] = True
        
        rating_match = re.search(r'rating (?:above|over|higher than) (\d+(?:\.\d+)?)', query_lower)
        if rating_match:
            features['rating_min'] = float(rating_match.group(1))
        
        for quality, keywords in self.rating_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                features['quality_preference'] = quality
                break
        
        year_match = re.search(r'(?:from|after|since) (\d{4})', query_lower)
        if year_match:
            features['year_min'] = int(year_match.group(1))
        
        year_range_match = re.search(r'(\d{4})(?:\s*-\s*|\s+to\s+)(\d{4})', query_lower)
        if year_range_match:
            features['year_min'] = int(year_range_match.group(1))
            features['year_max'] = int(year_range_match.group(2))
        
        return features
    
    def build_where_filter(self, features: Dict) -> Optional[Dict]:
        filters = []
        
        if features['genres']:
            genre_filters = []
            for genre in features['genres']:
                genre_filters.append({"genres": {"$contains": genre}})
            if len(genre_filters) == 1:
                filters.append(genre_filters[0])
            else:
                filters.append({"$or": genre_filters})
        
        if features['runtime_max']:
            filters.append({"runtime": {"$lte": features['runtime_max']}})
        
        if features['rating_min']:
            filters.append({"vote_average": {"$gte": features['rating_min']}})
        
        if features['year_min']:
            filters.append({"year": {"$gte": features['year_min']}})
        
        if features['year_max']:
            filters.append({"year": {"$lte": features['year_max']}})
        
        if features['quality_preference'] == 'high':
            filters.append({"vote_average": {"$gte": 7.5}})
        elif features['quality_preference'] == 'good':
            filters.append({"vote_average": {"$gte": 6.0}})
        
        if len(filters) == 0:
            return None
        elif len(filters) == 1:
            return filters[0]
        else:
            return {"$and": filters}
    
    def enhance_query_for_real_events(self, query: str) -> str:
        if any(keyword in query.lower() for keyword in self.real_events_keywords):
            enhanced_query = f"{query} true story based on real events biographical historical documentary factual"
            return enhanced_query
        return query
    
    def process_query(self, user_query: str, n_results: int = 15) -> Tuple[List[Dict], Dict]:
        features = self.extract_query_features(user_query)
        
        enhanced_query = self.enhance_query_for_real_events(user_query)
        
        query_embedding = self.openai_client.get_embedding(enhanced_query)
        if not query_embedding:
            return [], features
        
        where_filter = self.build_where_filter(features)
        
        search_results = self.vector_store.search_similar_movies(
            query_embedding=query_embedding,
            n_results=n_results,
            where_filter=where_filter
        )
        
        formatted_results = []
        if search_results and search_results.get('metadatas') and search_results['metadatas'][0]:
            for i, metadata in enumerate(search_results['metadatas'][0]):
                result = {
                    'metadata': metadata,
                    'document': search_results['documents'][0][i] if search_results.get('documents') else '',
                    'distance': search_results['distances'][0][i] if search_results.get('distances') else 0
                }
                formatted_results.append(result)
        
        if not formatted_results and where_filter:
            print("No results with filters, trying without filters...")
            search_results = self.vector_store.search_similar_movies(
                query_embedding=query_embedding,
                n_results=n_results,
                where_filter=None
            )
            
            if search_results and search_results.get('metadatas') and search_results['metadatas'][0]:
                for i, metadata in enumerate(search_results['metadatas'][0]):
                    result = {
                        'metadata': metadata,
                        'document': search_results['documents'][0][i] if search_results.get('documents') else '',
                        'distance': search_results['distances'][0][i] if search_results.get('distances') else 0
                    }
                    formatted_results.append(result)
        
        return formatted_results, features
    
    def generate_recommendation_response(self, user_query: str, similar_movies: List[Dict]) -> str:
        if not similar_movies:
            return "I couldn't find any movies matching your criteria. Please try a different search or check if the movie database has been populated."
        
        response = self.openai_client.get_movie_recommendations(user_query, similar_movies)
        return response
    
    def get_movie_recommendations(self, user_query: str) -> Dict:
        similar_movies, features = self.process_query(user_query)
        
        response_text = self.generate_recommendation_response(user_query, similar_movies)
        
        return {
            'response': response_text,
            'movies': similar_movies,
            'extracted_features': features,
            'total_results': len(similar_movies)
        }