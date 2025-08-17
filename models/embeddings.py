import openai
from typing import List, Dict, Optional
import numpy as np
import time
from config.settings import OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL

class OpenAIClient:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured")
        openai.api_key = OPENAI_API_KEY
        self.embedding_model = EMBEDDING_MODEL
        self.llm_model = LLM_MODEL
    
    def get_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        for attempt in range(max_retries):
            try:
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Error getting embedding (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            try:
                response = openai.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                if i + batch_size < len(texts):
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                for text in batch:
                    embedding = self.get_embedding(text)
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        embeddings.append([0] * 1536)
        
        return embeddings
    
    def generate_chat_response(self, 
                              messages: List[Dict[str, str]], 
                              max_tokens: int = 500,
                              temperature: float = 0.7) -> Optional[str]:
        try:
            response = openai.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating chat response: {e}")
            return None
    
    def create_movie_recommendation_prompt(self, 
                                         user_query: str, 
                                         similar_movies: List[Dict],
                                         context_limit: int = 10) -> str:
        movies_context = ""
        for i, movie in enumerate(similar_movies[:context_limit]):
            metadata = movie.get('metadata', {})
            document = movie.get('document', '')
            
            title = metadata.get('title', 'Unknown Title')
            year = metadata.get('year', metadata.get('release_date', 'Unknown Year'))
            genres = metadata.get('genres', '').replace('|', ', ')
            rating = metadata.get('vote_average', metadata.get('avg_rating', 'N/A'))
            runtime = metadata.get('runtime', 'Unknown')
            
            movies_context += f"""
Movie {i+1}: {title} ({year})
Genres: {genres}
Rating: {rating}/10
Runtime: {runtime} minutes
Description: {document[:200]}...
"""
        
        prompt = f"""Based on the user's request: "{user_query}"

Here are some relevant movies from our database:
{movies_context}

Please provide personalized movie recommendations that match the user's criteria. For each recommendation:
1. Explain why it matches their request
2. Mention key details (genre, year, rating, runtime)
3. Highlight what makes it special or noteworthy
4. If asking for movies "based on real events", prioritize those with true story elements

Format your response as a friendly, informative recommendation list. Be conversational and helpful."""

        return prompt
    
    def get_movie_recommendations(self, 
                                user_query: str, 
                                similar_movies: List[Dict]) -> str:
        prompt = self.create_movie_recommendation_prompt(user_query, similar_movies)
        
        messages = [
            {"role": "system", "content": "You are a knowledgeable movie recommendation assistant. Provide helpful, accurate, and engaging movie suggestions based on user preferences."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.generate_chat_response(messages, max_tokens=800, temperature=0.7)
        
        if not response:
            return "I apologize, but I'm having trouble generating recommendations right now. Please try again later."
        
        return response