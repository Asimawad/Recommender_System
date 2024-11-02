import os
import pickle
import numpy as np
import polars as pl
import pandas as pd

def create_movie_data(movies_df,movie_idx_map):
            all_genres = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
                        "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                        "Sci-Fi", "Thriller", "War", "Western", "(no genres listed)"]
            genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}

            # Step 2: Create a function to map each movie to its genre indices
            def get_genre_indices(genre_str):
                    genres = genre_str.split('|')
                    return [genre_to_idx[genre] for genre in genres if genre in genre_to_idx]

            # Apply this function to each movie in the movies_df
            movies_df['genre_indices'] = movies_df['genres'].apply(get_genre_indices)
            # Define maximum number of genres any movie has, for fixed-size array
            max_genres = movies_df['genre_indices'].apply(len).max()

            # Step 3: Prepare a 2D numpy array with genre indices
            num_movies = len(movie_idx_map)
            movies_genres_array = np.full((num_movies, max_genres), -1, dtype=np.int32)  # Fill with -1 for padding

            # Populate the array
            for movie_id, idx in movie_idx_map.items():
                genres = movies_df.loc[movies_df['movieId'] == movie_id, 'genre_indices'].values
                if len(genres) > 0:
                    genre_indices = genres[0]  # Get the list of genre indices for this movie
                    movies_genres_array[idx, :len(genre_indices)] = genre_indices

            # Now, `movies_genres_array` is a 2D numpy array with shape (num_movies, max_genres)
            # Each row contains the genre indices for the movie, padded with -1 if needed
            # Initialize an array of lists with 18 elements (one for each genre)

            # Initialize an empty list to hold arrays
            genre_to_movies = [[] for _ in range(19)]

            # Populate the array
            for movie_idx, genres in enumerate(movies_genres_array):
                for genre in genres:

                    pure_g = [x for x in genres if x != -1 and x != genre ]
                    pure = np.array(pure_g,dtype = np.int32)

                    genre_to_movies[genre].append((movie_idx,pure))

            return movies_genres_array,genre_to_movies,genre_to_idx

def create_train_test_df(ratings):
    # The Data Structure
    grouped_users = ratings.group_by('userId', maintain_order=True).agg(pl.col('movieId'), pl.col('rating'))
    grouped_movies= ratings.group_by('movieId', maintain_order=True).agg(pl.col('userId') , pl.col('rating'))
    user_idx_map = {user_id: idx for idx, user_id in enumerate(grouped_users['userId'].to_list())}
    movie_idx_map = {movie_id: idx for idx, movie_id in enumerate(grouped_movies['movieId'].to_list())}
    del grouped_users , grouped_movies
 
    # Create inverse mapping for user indices
    idx_to_user = {idx: user_id for user_id, idx in user_idx_map.items()}

    # Create inverse mapping for movie indices
    idx_to_movie = {idx: movie_id for movie_id, idx in movie_idx_map.items()}

    def indexing_users(user_id):
        return user_idx_map[user_id]

    def indexing_movies(movie_id):
        return movie_idx_map[movie_id]

    # Add a column using a custom function
    ratings = ratings.with_columns([
        pl.col("userId").cast(pl.Int32),
        pl.col("movieId").cast(pl.Int32),
        pl.col("rating").cast(pl.Float32),
        pl.col("userId").map_elements(indexing_users, return_dtype = pl.Int32).alias("userIdx"),
        pl.col("movieId").map_elements(indexing_movies,return_dtype = pl.Int32).alias("movieIdx"),])


    # method 2 - very good
    def train_test_split_df(df, seed=0, test_size=0.2):
        return df.with_columns(
            pl.int_range(pl.len(), dtype=pl.UInt32)
            .shuffle(seed=seed)
            .gt(pl.len() * test_size)
            .alias("split")
        ).partition_by("split", include_key=False)


    def train_test_split(X,seed=0, test_size=0.1):
        (X_train, X_test) = train_test_split_df(X, seed=seed, test_size=test_size)
        return (X_train, X_test)

    train_df,test_df = train_test_split(ratings)
    return  train_df,test_df,user_idx_map,  movie_idx_map ,idx_to_user, idx_to_movie

def split(dataframe):

  grouped_users = dataframe.group_by('userIdx', maintain_order=True).agg(pl.col('movieIdx'), pl.col('rating'))
  users_data = grouped_users[:,1:].to_numpy()
  user_indices = grouped_users[:,0].to_numpy()

  grouped_movies= dataframe.group_by('movieIdx', maintain_order=True).agg(pl.col('userIdx') , pl.col('rating'))
  movies_data = grouped_movies[:,1:].to_numpy()
  movies_indices = grouped_movies[:,0].to_numpy()

  return users_data,user_indices,movies_data,movies_indices

def save_idx_maps(experiment_folder,user_idx_map,  movie_idx_map ,idx_to_user,idx_to_movie,genre_to_idx,genre_to_movies):
    # Create the directory if it doesn't exist
    os.makedirs(experiment_folder, exist_ok=True)

    # Save the dictionaries
    with open(os.path.join(experiment_folder, 'user_idx_map.pkl'), 'wb') as f:
        pickle.dump(user_idx_map, f)

    with open(os.path.join(experiment_folder, 'movie_idx_map.pkl'), 'wb') as f:
        pickle.dump(movie_idx_map, f)

    with open(os.path.join(experiment_folder, 'idx_to_user.pkl'), 'wb') as f:
        pickle.dump(idx_to_user, f)

    with open(os.path.join(experiment_folder, 'idx_to_movie.pkl'), 'wb') as f:
        pickle.dump(idx_to_movie, f)

    with open(os.path.join(experiment_folder, 'genre_to_idx.pkl'), 'wb') as f:
        pickle.dump(genre_to_idx, f)

    with open(os.path.join(experiment_folder, 'genre_to_movies.pkl'), 'wb') as f:
        pickle.dump(genre_to_movies, f)
    print("Dictionaries saved successfully!")

def Save_data_split(experiment_folder,train_df,test_df,users_train,users_test,users_test_idxes,movies_test_idxes,users_train_idxes,movies_train_idxes,movies_train,movies_test,movies_genres_array):
    os.makedirs(experiment_folder, exist_ok=True)
    np.save(f'./{experiment_folder}/users_train.npy', users_train,allow_pickle=True)
    np.save(f'./{experiment_folder}/users_test.npy', users_test,allow_pickle=True)

    np.save(f'./{experiment_folder}/users_test_idxes.npy', users_test_idxes,allow_pickle=True)
    np.save(f'./{experiment_folder}/movies_test_idxes.npy', movies_test_idxes,allow_pickle=True)

    np.save(f'./{experiment_folder}/users_train_idxes.npy', users_train_idxes,allow_pickle=True)
    np.save(f'./{experiment_folder}/movies_train_idxes.npy', movies_train_idxes,allow_pickle=True)

    np.save(f'./{experiment_folder}/movies_train.npy', movies_train,allow_pickle=True)
    np.save(f'./{experiment_folder}/movies_test.npy', movies_test,allow_pickle=True)


    np.save(f'./{experiment_folder}/movies_genres_array.npy', movies_genres_array,allow_pickle=True)


    train_df.write_csv(f'./{experiment_folder}/train_df.csv' ,separator  = "," )
    test_df.write_csv(f'./{experiment_folder}/test_df.csv', separator = "," )

ratings = pl.read_csv("./ml-32m/ratings.csv")
movies_df = pd.read_csv("./ml-32m/movies.csv")
data_folder="TRAIN_TEST_DATA"
train_df,test_df,user_idx_map,movie_idx_map ,idx_to_user, idx_to_movie = create_train_test_df(ratings)
users_test,users_test_idxes,movies_test,movies_test_idxes = split(test_df)
users_train,users_train_idxes,movies_train,movies_train_idxes = split(train_df)
movies_genres_array,genre_to_movies,genre_to_idx = create_movie_data(movies_df,movie_idx_map)
Save_data_split(data_folder,train_df,test_df,users_train,users_test,users_test_idxes,movies_test_idxes,users_train_idxes,movies_train_idxes,movies_train,movies_test,movies_genres_array)
save_idx_maps(data_folder,user_idx_map,  movie_idx_map ,idx_to_user,idx_to_movie,genre_to_idx,genre_to_movies)
