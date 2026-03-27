# Movie Recommender System

A content-based movie recommendation engine that uses TF-IDF vectorization and cosine similarity to suggest movies based on descriptions and genres.

## Setup Instructions

### Prerequisites
- Python 3.7+
- pip 

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sarajdhakal/movie_recommender_system.git
cd movie_recommender_system
```
2. Install required dependencies:
```bash
pip install numpy pandas scikit-learn nltk matplotlib
```
3. Download NLTK data:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```
4. Run the Jupyter Notebook:
```bash
jupyter notebook movie_recommender.ipynb
```

## Project Overview

This project implements a content-based recommendation system that recommends movies similar to a given movie based on:

- Movie descriptions
- Genre information
- Combined features (tags)

Dataset: IMDB Genres dataset from hugging face (238,256 movies initially, reduced to 168,533 after cleaning)

### Key Features

- Search by movie title
- Search by description/genre keywords
- Top-5 movie recommendations
- TF-IDF vectorization for text representation

## Approach & Methodology

### Content-Based Filtering

The system uses content-based filtering:

1. Extract movie features (description + genres)
2. Convert text to numerical vectors (TF-IDF)
3. Calculate similarity between movies (cosine similarity)
4. Recommend top-K most similar movies

### Architecture Flow

- Raw Data
- Data Cleaning (null, duplicates)
- Text Preprocessing (tokenization, lowercase, stemming)
- Feature Engineering (tags = description + genres)
- TF-IDF Vectorization (3000 features)
- Cosine Similarity Matrix (5000 × 5000)
- Recommendation Engine
- Top-5 Similar Movies

## Limitations

1. Cold Start Problem
   - New movies without historical data cannot be recommended
   - New users' preferences are unknown
2. No Collaborative Filtering
   - Ignores user-user and item-item interactions
   - Cannot leverage crowd wisdom
3. Missing User Data
   - No user ratings or watch history
   - Cannot personalize recommendations based on user preferences
   - Cannot perform user-based or hybrid recommendations
4. Limited Dataset Size
   - Using only 5,000 movies from 168,533 available
   - Reduced due to computational constraints
   - Smaller database = fewer diverse recommendations
5. Shallow Text Understanding
   - TF-IDF treats words independently
   - No understanding of semantic meaning or context
   - "Excellent movie" vs "movie excellent" treated the same

## Future Improvements

- Add collaborative filtering (user-user, item-item)
- Implement hybrid recommendation model (content + collaborative)
- Add user profiles, ratings, history, and personalization
- Use word embeddings (Word2Vec/BERT) for semantic similarity
- Add temporal decay to prefer newer releases
- Include diversity and novelty constraints
- Build a web UI or API for live queries

## Evaluation Metrics
- Cosine Similarity Score

