import imdb
from sentence_transformers import SentenceTransformer
import AI.synopsis_gen as sgen
import numpy as np
import time
from functools import lru_cache

# Create IMDb and SentenceTransformer once
ia = imdb.IMDb()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Cache embeddings to avoid repeated API calls
@lru_cache(maxsize=100)
def get_movie_synopsis_embedding(movie_name):
    """
    Returns embedding vector for movie synopsis
    """
    print(f'Fetching movie synopsis for {movie_name}...')
    try:
        results = ia.search_movie(movie_name)
        if not results:
            return None

        movie = results[0]

        # Only request the synopsis info (faster than full update)
        ia.update(movie, info=['synopsis'])

        synopsis_list = movie.get('synopsis', [])
        if not synopsis_list:
            try:
                synopsis = sgen.ask_gemini(f'Give me synopsis of this movie or series under 25 lines, Just give me the synopsis nothing else: {movie_name}')
            except:
                synopsis = f"A movie titled {movie_name}"

        else:
            synopsis = synopsis_list[0]

        # Create embedding
        embedding = embedder.encode(synopsis)
        print(f'Synopsis for {movie_name} fetched successfully.')
        print('##'*20)
        return embedding

    except Exception as e:
        print(f"An error occurred: {e}")
        # Return a default embedding in case of error
        return embedder.encode(f"Movie: {movie_name}")

# Get movie summary short one from gemini with caching
@lru_cache(maxsize=100)
def get_movie_summary_embedding(movie_name):
    try:
        summary = sgen.ask_gemini(f'Give me summary of this movie or series under 3 sentences, include genre names: {movie_name}')

        # Create embedding
        embedding = embedder.encode(summary)
        print(f'Summary for {movie_name} fetched successfully. ')
        print('#'*20)
        return embedding
    
    except Exception as e: 
        print(f'An error occurred: {e}')
        # Return a default embedding in case of error
        return embedder.encode(f"Movie: {movie_name}")

# Fallback function if Gemini fails
def get_fallback_embedding(movie_name):
    """Generate a simple embedding based on movie name"""
    return embedder.encode(f"Movie film {movie_name}")

# Main function that tries multiple approaches
def get_movie_embedding(movie_name, use_synopsis=False):
    """
    Get embedding for a movie with fallback mechanisms
    """
    try:
        if use_synopsis:
            embedding = get_movie_synopsis_embedding(movie_name)
        else:
            embedding = get_movie_summary_embedding(movie_name)
        
        if embedding is None or len(embedding) == 0:
            return get_fallback_embedding(movie_name)
        
        return embedding
        
    except Exception as e:
        print(f"Failed to get embedding for {movie_name}: {e}")
        return get_fallback_embedding(movie_name)

# Batch processing for multiple movies
def get_movie_embeddings_batch(movie_names, use_synopsis=False):
    """
    Get embeddings for multiple movies efficiently
    """
    embeddings = []
    for movie_name in movie_names:
        embedding = get_movie_embedding(movie_name, use_synopsis)
        embeddings.append(embedding)
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    return np.array(embeddings)

# Example usage
if __name__ == "__main__":
    movie_name = 'Now You See Me'  # Example movie name
    synopsis_embedding = get_movie_summary_embedding(movie_name)

    if synopsis_embedding is not None:
        print("\nEmbedding vector (first 5 values):")
        print(synopsis_embedding[:5])
        print(f"Embedding shape: {synopsis_embedding.shape}")