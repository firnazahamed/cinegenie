#!/usr/bin/env python3

import os
import sys
import streamlit as st

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cinegenie import SimpleMovieRecommendationSystem

def main():
    st.set_page_config(
        page_title="🎬 CineGenie - AI Movie Recommendations",
        page_icon="🎬",
        layout="wide"
    )
    
    # Custom CSS for full background gradient
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: white !important;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stMarkdown p {
        color: white !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stMarkdown h3 {
        color: white !important;
    }
    .examples-section {
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    .examples-title {
        color: white !important;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .stTextInput label {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎬 CineGenie - AI Movie Recommendations")
    st.write("Get personalized movie recommendations using AI-powered semantic search!")
    
    # Initialize system
    if 'movie_system' not in st.session_state:
        st.session_state.movie_system = SimpleMovieRecommendationSystem()
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    tmdb_key = os.getenv("TMDB_API_KEY")
    
    if not openai_key or not tmdb_key:
        st.error("⚠️ Please configure your API keys in the .env file")
        st.stop()
    
    # Search input with button on the right
    st.markdown("### 💬 What kind of movie are you looking for?")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your movie request:",
            placeholder="Find me a horror movie based on real events with runtime under 2 hours",
            help="Describe what you're looking for. You can mention genres, themes, ratings, or specific preferences.",
            label_visibility="collapsed"
        )
    
    with col2:
        st.write("")  # Add some spacing
        search_button = st.button("🔍 Search Movies", type="primary")
    
    # Try these examples section
    st.markdown('<div class="examples-section">', unsafe_allow_html=True)
    st.markdown('<p class="examples-title">💡 Try these examples:</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎭 Horror Movies", key="horror"):
            st.session_state.example_query = "popular horror movies with high ratings"
            st.rerun()
    
    with col2:
        if st.button("⭐ Top Rated Movies", key="top_rated"):
            st.session_state.example_query = "highly rated movies from recent years"
            st.rerun()
    
    with col3:
        if st.button("🤣 Comedy Movies", key="comedy"):
            st.session_state.example_query = "funny comedy movies that will make me laugh"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle example queries
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
        search_button = True
    
    if search_button and query:
        with st.spinner("Searching for movies..."):
            try:
                results = st.session_state.movie_system.get_recommendations(query, use_embeddings=True)
                
                if results['movies']:
                    st.markdown("---")
                    st.markdown("### 🤖 AI Recommendation")
                    st.write(results['recommendation'])
                    
                    st.markdown("### 🎬 Movies Found")
                    for i, movie in enumerate(results['movies'][:5], 1):
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                title = movie.get('title', 'Unknown')
                                year = movie.get('release_date', 'Unknown')[:4] if movie.get('release_date') else 'Unknown'
                                rating = movie.get('vote_average', 'N/A')
                                overview = movie.get('overview', 'No description available')
                                
                                st.write(f"**{i}. {title} ({year})**")
                                st.write(f"Rating: {rating}/10")
                                st.write(overview)
                            
                            with col2:
                                similarity = movie.get('similarity_score', 0)
                                st.metric("Match", f"{similarity:.2f}")
                            
                            st.markdown("---")
                else:
                    st.warning("No movies found. Try a different search.")
                    
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()