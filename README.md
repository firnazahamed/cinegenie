# 🎬 CineGenie - AI-Powered Movie Recommendation Chatbot

CineGenie is an intelligent movie recommendation system that uses RAG (Retrieval-Augmented Generation) to provide personalized movie suggestions based on natural language queries.

## 🌟 Features

- **Natural Language Queries**: Ask for movies in plain English
- **Intelligent Filtering**: Automatically extracts preferences (genre, runtime, rating, year)
- **Real Events Detection**: Identifies requests for movies based on true stories
- **Semantic Search**: Uses OpenAI embeddings for similarity matching
- **Rich Movie Data**: Combines TMDB current movies with MovieLens historical ratings
- **Interactive Web Interface**: Clean Streamlit-based chat interface

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: OpenAI text-embedding-ada-002
- **LLM**: OpenAI GPT-3.5-turbo
- **Data Sources**: TMDB API + MovieLens dataset
- **Backend**: Python

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- TMDB API key

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd cinegenie
pip install -r requirements.txt
```

### 2. Configuration

Create your `.env` file with API keys:

```bash
# Create .env file with your API keys
OPENAI_API_KEY=your_openai_api_key_here
TMDB_API_KEY=your_tmdb_api_key_here
TMDB_BASE_URL=https://api.themoviedb.org/3
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-3.5-turbo
```

### 3. Get API Keys

**OpenAI API Key:**
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add it to your `.env` file

**TMDB API Key:**
1. Visit [TMDB API](https://www.themoviedb.org/settings/api)
2. Request an API key (free)
3. Add it to your `.env` file

### 4. Run the Application

```bash
# Option 1: Using the launcher (recommended)
python run.py

# Option 2: Direct streamlit command
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

## 🎯 Example Queries

- "Find me a horror movie which is based on real events with runtime under 2 hours"
- "Recommend recent sci-fi movies with high ratings"
- "Show me romantic comedies from the 2010s"
- "Find action movies similar to Mad Max"

## 🧪 Testing

Test the system with different commands:

```bash
# Test the core system
python cinegenie.py
```

## 📁 Project Structure

```
cinegenie/
├── app/
│   └── streamlit_app.py      # Streamlit web interface
├── config/
│   └── settings.py           # Configuration settings
├── data/
│   ├── tmdb_client.py        # TMDB API integration
│   └── movielens_client.py   # MovieLens data processing
├── models/
│   ├── embeddings.py         # OpenAI integration
│   └── vector_store.py       # ChromaDB vector database
├── utils/
│   ├── data_processor.py     # Data preprocessing pipeline
│   └── query_processor.py    # RAG query processing
├── cinegenie.py              # Core recommendation system
├── requirements.txt          # Python dependencies
├── run.py                   # Streamlit launcher
└── README.md               # This file
```

## 🔧 How It Works

1. **Data Collection**: Fetches movies from TMDB API and ratings from MovieLens
2. **Preprocessing**: Creates rich movie descriptions combining metadata and plot
3. **Embedding Generation**: Uses OpenAI to create vector embeddings for semantic search
4. **Vector Storage**: Stores embeddings in ChromaDB for fast similarity search
5. **Query Processing**: 
   - Extracts features from natural language queries
   - Generates query embeddings
   - Performs similarity search with filters
   - Uses GPT to generate personalized recommendations

## ⚙️ Configuration Options

### Database Settings
- `CHROMA_PERSIST_DIRECTORY`: Where to store the vector database
- Max movies to process (configurable in data processor)

### Model Settings
- `EMBEDDING_MODEL`: OpenAI embedding model (default: text-embedding-ada-002)
- `LLM_MODEL`: Language model for responses (default: gpt-3.5-turbo)

## 🔍 Query Features

The system automatically detects:
- **Genres**: horror, comedy, action, drama, etc.
- **Runtime limits**: "under 2 hours", "less than 90 minutes"
- **Ratings**: "highly rated", "rating above 7"
- **Real events**: "based on true story", "real events"
- **Years**: "from 2010", "2000s movies"
- **Quality preferences**: "top rated", "best", "recent"

## 🎨 Web Interface Features

- Chat-style query input
- AI-generated recommendations
- Detailed movie cards with metadata
- Similarity scores
- Feature extraction display
- Database management tools
- Quick action buttons

## 🚨 Troubleshooting

**No movies found:**
- Ensure database is populated (`python test_query.py` to check)
- Verify API keys are correct
- Try broader search queries

**API Errors:**
- Check API key validity
- Verify internet connection
- Check API rate limits

**Embedding Errors:**
- Ensure OpenAI API key has sufficient credits
- Check for API rate limiting

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📧 Support

For issues and questions, please create an issue in the repository.