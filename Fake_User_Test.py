import numpy as np
import pandas as pd
import Helper_Functions
from Helper_Functions import load_model, Load_idx_maps

# Define constants
DATA_FOLDER = "TRAIN_TEST_DATA"
EXPERIMENT_FOLDER = "Experiments/32_million_trials/"
MOVIE_FILE_PATH = './ml-32m/movies.csv'

K_FACTORS = 30
LAMBDA_REG = 1
GAMMA = 0.001
TAW = 10

# Function to load data
def load_data():
    movies = pd.read_csv(MOVIE_FILE_PATH)
    movies_factors, _, _, item_bias = load_model(EXPERIMENT_FOLDER)
    _, movie_idx_map, _, idx_to_movie, _, _ = Load_idx_maps(DATA_FOLDER)
    return movies, movies_factors, item_bias, movie_idx_map, idx_to_movie

# Function to get movie details
def get_movie_details(movies, title):
    filtered_movie = movies[movies['title'] == title]
    if not filtered_movie.empty:
        movie_name = filtered_movie['title'].values[0]
        movie_id = filtered_movie['movieId'].values[0]
        genre = filtered_movie['genres'].values[0]
        return movie_name, movie_id, genre
    else:
        return None, None, None

# Function to create a fake user
def create_fake_user(movie_idx_map, movieId, item_bias, movies_factors):
    movie_idx = movie_idx_map[movieId.item()]
    list_of_favourite_movies = [movie_idx, 5]
    return Helper_Functions.create_fake_user(
        list_of_favourite_movies, item_bias, movies_factors, K_FACTORS, LAMBDA_REG, GAMMA, TAW
    )

# Function to generate movie recommendations
def generate_recommendations(asim, movies_factors, item_bias, idx_to_movie, movies, movie_name):
    recommendation_scores = (movies_factors @ asim) + (item_bias * 0.05)
    top_movie_indices = np.argsort(recommendation_scores)[::-1][:50]
    top_movie_ids = [idx_to_movie[idx] for idx in top_movie_indices]

    recommendations = {'title': [], 'genre': []}
    for MID in top_movie_ids:
        filtered_movie = movies[movies['movieId'] == MID]
        if not filtered_movie.empty:
            movieId, title, genre = (
                filtered_movie['movieId'].values[0],
                filtered_movie['title'].values[0],
                filtered_movie['genres'].values[0]
            )
            if title != movie_name:
                recommendations['title'].append(title)
                recommendations['genre'].append(genre)
    return pd.DataFrame(recommendations)

# Main function
def main():
    # Load data
    movies, movies_factors, item_bias, movie_idx_map, idx_to_movie = load_data()

    # Get movie details
    movie_name, movie_id, genre = get_movie_details(movies, "Toy Story (1995)")
    if movie_name is None:
        print("Movie not found.")
        return

    print(f"The user liked the movie: {movie_name}, with id: {movie_id}, The genre is {genre}")

    # Create a fake user
    asim = create_fake_user(movie_idx_map, movie_id, item_bias, movies_factors)

    # Generate recommendations
    top_movies = generate_recommendations(asim, movies_factors, item_bias, idx_to_movie, movies, movie_name)

    # Print top recommendations
    print("Recommendations are:\n")
    print(top_movies.head(25))

# Run the main function
if __name__ == "__main__":
    main()
