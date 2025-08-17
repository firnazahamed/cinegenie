#!/usr/bin/env python3

import os
import sys
import requests
import json
import pickle
import time
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()


class SimpleMovieRecommendationSystem:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.tmdb_api_key = os.getenv("TMDB_API_KEY")
        self.movies_cache = []
        self.embeddings_cache = {}

    def get_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from OpenAI API"""

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        data = {"model": "text-embedding-ada-002", "input": text}

        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=data,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
            else:
                print(f"Embedding API error: {response.status_code}")
                return None

        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def get_openai_recommendation(self, query: str, movies: List[Dict]) -> str:
        """Get movie recommendation from OpenAI"""

        # Create context from movies
        movie_context = ""
        for i, movie in enumerate(movies[:5], 1):
            title = movie.get("title", "Unknown")
            year = (
                movie.get("release_date", "Unknown")[:4]
                if movie.get("release_date")
                else "Unknown"
            )
            rating = movie.get("vote_average", "N/A")
            overview = movie.get("overview", "No description")[:200]
            runtime = movie.get("runtime", "Unknown")

            movie_context += f"""
{i}. {title} ({year}) - Rating: {rating}/10
   Runtime: {runtime} minutes
   Description: {overview}...
"""

        prompt = f"""Based on the user's request: "{query}"

Here are the ONLY movies you should recommend from (DO NOT suggest any other movies):
{movie_context}

IMPORTANT: You must ONLY recommend movies from the list above. Do not suggest any movies not in this list.

Please provide 2-3 personalized movie recommendations that best match the user's criteria from the movies listed above. For each recommendation:
1. Explain why it matches their request
2. Mention key details (genre, year, rating, runtime) 
3. Highlight what makes it special
4. Reference the exact title and year as shown in the list

Be conversational and helpful, but stick strictly to the provided movies."""

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.7,
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"Chat API error: {response.status_code}")
                return "Error generating recommendations"

        except Exception as e:
            print(f"Chat error: {e}")
            return "Error generating recommendations"

    def fetch_movies_from_tmdb(self) -> List[Dict]:
        """Fetch movies from TMDB API"""

        print("Fetching movies from TMDB...")

        movies = []
        endpoints = [
            f"https://api.themoviedb.org/3/movie/popular?api_key={self.tmdb_api_key}&page=1",
            f"https://api.themoviedb.org/3/movie/top_rated?api_key={self.tmdb_api_key}&page=1",
            f"https://api.themoviedb.org/3/discover/movie?api_key={self.tmdb_api_key}&with_genres=27&sort_by=popularity.desc&page=1",  # Horror
            f"https://api.themoviedb.org/3/discover/movie?api_key={self.tmdb_api_key}&with_keywords=9714&sort_by=popularity.desc&page=1",  # True story
        ]

        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    movies.extend(data.get("results", []))
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"Error fetching movies: {e}")

        # Remove duplicates
        unique_movies = {movie["id"]: movie for movie in movies}
        self.movies_cache = list(unique_movies.values())

        print(f"Loaded {len(self.movies_cache)} unique movies")
        return self.movies_cache

    def filter_movies_by_query(self, query: str, movies: List[Dict]) -> List[Dict]:
        """Filter movies based on query criteria"""

        query_lower = query.lower()
        filtered = []

        for movie in movies:
            score = 0
            title = movie.get("title", "").lower()
            overview = movie.get("overview", "").lower()

            # Genre filtering
            if "horror" in query_lower and (
                "horror" in overview or "scary" in overview
            ):
                score += 3
            if "comedy" in query_lower and "comedy" in overview:
                score += 3
            if "action" in query_lower and "action" in overview:
                score += 3

            # Real events filtering
            if any(
                keyword in query_lower
                for keyword in ["real events", "true story", "based on"]
            ):
                if any(keyword in overview for keyword in ["true", "real", "based"]):
                    score += 3

            # Rating filtering
            if "highly rated" in query_lower or "top rated" in query_lower:
                rating = movie.get("vote_average", 0)
                if rating >= 7.5:
                    score += 2

            # General relevance
            query_words = query_lower.split()
            for word in query_words:
                if word in title:
                    score += 2
                if word in overview:
                    score += 1

            if score > 0:
                movie["relevance_score"] = score
                filtered.append(movie)

        # Sort by relevance and rating
        filtered.sort(
            key=lambda x: (x.get("relevance_score", 0), x.get("vote_average", 0)),
            reverse=True,
        )

        return filtered[:10]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return dot_product / (magnitude1 * magnitude2)

    def semantic_search(
        self, query: str, movies: List[Dict], limit: int = 10
    ) -> List[Dict]:
        """Perform semantic search using embeddings"""

        print("Performing semantic search...")

        # Get query embedding
        query_embedding = self.get_openai_embedding(query)
        if not query_embedding:
            print("Could not get query embedding, falling back to keyword search")
            return self.filter_movies_by_query(query, movies)

        movie_similarities = []

        for movie in movies:
            # Create movie description
            description = f"{movie.get('title', '')} {movie.get('overview', '')}"

            # Get or compute movie embedding
            movie_id = movie.get("id")
            if movie_id in self.embeddings_cache:
                movie_embedding = self.embeddings_cache[movie_id]
            else:
                movie_embedding = self.get_openai_embedding(description)
                if movie_embedding:
                    self.embeddings_cache[movie_id] = movie_embedding
                else:
                    continue

            # Calculate similarity
            similarity = self.cosine_similarity(query_embedding, movie_embedding)
            movie["similarity_score"] = similarity
            movie_similarities.append(movie)

            time.sleep(0.1)  # Rate limiting

        # Sort by similarity
        movie_similarities.sort(
            key=lambda x: x.get("similarity_score", 0), reverse=True
        )

        return movie_similarities[:limit]

    def get_recommendations(self, query: str, use_embeddings: bool = True) -> Dict:
        """Get movie recommendations for a query"""

        print(f"\n Processing query: '{query}'")
        print("=" * 60)

        if not self.movies_cache:
            self.fetch_movies_from_tmdb()

        if use_embeddings:
            relevant_movies = self.semantic_search(query, self.movies_cache)
        else:
            relevant_movies = self.filter_movies_by_query(query, self.movies_cache)

        if not relevant_movies:
            return {
                "query": query,
                "movies": [],
                "recommendation": "No movies found matching your criteria.",
            }

        print(f"Found {len(relevant_movies)} relevant movies")

        # Get AI recommendation
        print("🤖 Generating AI recommendation...")
        ai_recommendation = self.get_openai_recommendation(query, relevant_movies)

        return {
            "query": query,
            "movies": relevant_movies,
            "recommendation": ai_recommendation,
        }

    def display_results(self, results: Dict):
        """Display recommendation results"""

        print(f"\nResults for: '{results['query']}'")
        print("=" * 60)

        print(f"\n🤖 AI Recommendation:")
        print(results["recommendation"])

        print(f"\n📋 Matching Movies:")
        for i, movie in enumerate(results["movies"][:5], 1):
            title = movie.get("title", "Unknown")
            year = (
                movie.get("release_date", "Unknown")[:4]
                if movie.get("release_date")
                else "Unknown"
            )
            rating = movie.get("vote_average", "N/A")
            similarity = movie.get("similarity_score", movie.get("relevance_score", 0))

            print(f"{i}. {title} ({year}) - {rating}/10")
            print(f"   Match Score: {similarity:.3f}")
            print(f"   Plot: {movie.get('overview', 'No description')[:100]}...")
            print()


def main():
    print("CineGenie - AI Movie Recommendation System")
    print("=" * 50)

    system = SimpleMovieRecommendationSystem()

    # Test queries
    test_queries = [
        "Find me a horror movie which is based on real events with runtime under 2 hours",
        "Recommend highly rated science fiction movies",
        "Show me funny romantic comedies",
    ]

    for query in test_queries:
        try:
            results = system.get_recommendations(query, use_embeddings=True)
            system.display_results(results)

            print("\n" + "=" * 60 + "\n")

        except Exception as e:
            print(f"Error processing query '{query}': {e}")

    print("Demo complete!")
    print("Your movie recommendation system is working with:")
    print("OpenAI embeddings for semantic search")
    print("GPT-3.5 for intelligent recommendations")
    print("TMDB movie database")
    print("Natural language query processing")


if __name__ == "__main__":
    main()
