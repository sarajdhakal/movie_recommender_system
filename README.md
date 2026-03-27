# Movie Recommender System

A content-based movie recommendation engine that uses TF-IDF vectorization and cosine similarity to suggest movies based on descriptions and genres.

## Table of Contents
- [Setup Instructions](#setup-instructions)
- [Project Overview](#project-overview)
- [Approach & Methodology](#approach--methodology)
- [Data Preprocessing](#data-preprocessing)
- [Vectorization & Similarity](#vectorization--similarity)
- [Usage Examples](#usage-examples)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Evaluation Metrics](#evaluation-metrics)

## Setup Instructions

### Prerequisites
- Python 3.7+
- pip or conda

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

Dataset: IMDB Genres dataset (238,256 movies initially, reduced to 168,533 after cleaning)

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

## Data Preprocessing

1. Data Loading & Exploration

```python
movies = pd.read_csv("hf://datasets/jquigl/imdb-genres/train.csv")
# Initial shape: (238,256, 5)
# Columns: movie title - year, genre, expanded-genres, rating, description
```

2. Missing Value Handling

- Rating column: 69,721 null values (removed)
- Final dataset: 168,535 rows

```python
movies.dropna(inplace=True)
```

3. Duplicate Removal

- Found 2 duplicate entries
- Removed using `drop_duplicates()`
- Final dataset: 168,533 unique movies

4. Text Tokenization

```python
movies['clean-description'] = movies['description'].apply(lambda x: x.split())
movies['expanded-genres'] = movies['expanded-genres'].apply(lambda x: x.split())
```

5. Feature Engineering

```python
movies['tags'] = movies['clean-description'] + movies['expanded-genres']
```

6. Text Normalization

```python
# Convert to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Porter Stemming (reduces words to root form)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stems(text):
    T = []
    for i in text.split():
        T.append(ps.stem(i))
    return " ".join(T)

new_df['tags'] = new_df['tags'].apply(stems)
```

### Example Preprocessing Output

Movie | Tags
--- | ---
Flaming Ears - 1992 | flaming ear pop sci-fi lesbian fantasi featur set year 2700 fictiv burn-out citi asche follow tangl life three women volle nun spy fantasi sci-fi
Jeg elsker dig - 1957 | six peopl three coupl meet random atmospher come togeth night drunk love romance drama comedi

## Vectorization & Similarity

### TF-IDF Vectorization

Why TF-IDF?

- Term Frequency (TF): How often a word appears in a document
- Inverse Document Frequency (IDF): How unique/important a word is across all documents
- Reduces impact of common stopwords ("the", "is", "and")
- Gives higher weight to distinctive movie features

Configuration:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer(max_features=3000, stop_words='english')
transformed_output = v.fit_transform(new_df['tags']).toarray()
# Output shape: (5000, 3000) - 5000 movies, 3000 features
```

Parameters:

- `max_features=3000`: Keep top 3000 most important terms
- `stop_words='english'`: Remove common English words
- Vocabulary size: 3000 unique terms

### Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(transformed_output)
# Output shape: (5000, 5000)
# similarity[i][j] = cosine similarity between movie i and j
```

Similarity Range: 0 to 1

- 1.0 = Identical movies
- 0.0 = Completely different movies

## Usage Examples

### Example 1: Recommend by Movie Title

```python
def recommend(movie):
    movie = str(movie).lower()
    matches = new_df[new_df['movie title - year'].str.lower().str.contains(movie)]

    if matches.empty:
        print("No movie found")
        return

    idx = matches.index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

    print(f"Best match is: {new_df.iloc[idx]['movie title - year']}")
    print("Recommendations are:")
    for i in distances[1:6]:
        print(new_df.iloc[i[0]]['movie title - year'])

# Usage
recommend('Alien')
```

Sample Output:

- Best match is: Alien Presence - 2009
- Alien 3 - 1992
- Alien Resurrection - 1997
- Alien Species - 2007
- Aliens vs. Predator: Requiem - 2007
- Creatures from the Abyss - 1994

### Example 2: Search by Description/Genre

```python
def search_by_description(description):
    description_vector = v.transform([description.lower()])
    sim_scores = cosine_similarity(description_vector, transformed_output)
    distances = sorted(list(enumerate(sim_scores[0])), reverse=True, key=lambda x: x[1])

    print(f"\nTop matches for: '{description}'")
    print("-" * 30)
    for i in distances[0:5]:
        print(f"{new_df.iloc[i[0]]['movie title - year']}")

# Usage
search_by_description("horror")
search_by_description("young gang member life around city")
```

Sample Output:

- The Shining - 1980
- Insidious - 2010
- Sinister - 2012
- Paranormal Activity - 2007
- The Ring - 2002

- Menace II Society - 1993
- Boyz n the Hood - 1991
- Training Day - 2001
- American Gangster - 2007
- Juice - 1992

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
6. Genre-Dominated Features
   - Genre tags (short) + descriptions (long)
   - Genre information may be underweighted
   - Descriptions dominate the similarity calculation
7. No Temporal Information
   - Doesn't consider movie release dates
   - Cannot recommend new movies preferentially
   - Ignores time-based user preferences
8. Static Recommendations
   - Same recommendations for same query
   - No diversity or serendipity
   - Cannot handle cold-start items

## Future Improvements

- Add collaborative filtering (user-user, item-item)
- Implement hybrid recommendation model (content + collaborative)
- Add user profiles, ratings, history, and personalization
- Use word embeddings (Word2Vec/BERT) for semantic similarity
- Add temporal decay to prefer newer releases
- Include diversity and novelty constraints
- Build a web UI or API for live queries

## Evaluation Metrics

- Precision@K
- Recall@K
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)
- Coverage
- Diversity
