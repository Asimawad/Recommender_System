import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import polars as pl
import os
# The data Structure + Helper_Functions 
import Helper_Functions
# experiment_folder = "Experiments/32_million_trials/K_30_n_Epochs_40_lmbd_1_gamma_0.1_taw_10/"
experiment_folder = "Experiments/32_million_trials/"
# moviess = pl.read_csv(f'ml-32m/movies.csv')
# ratings = pl.read_csv('ml-32m/ratings.csv')
K_factors = 30; lambda_reg = 1 ; gamma = 1 ; taw =  10 
# # K_factors = 25 ; n_Epochs = 40 ;  lambda_reg = 20 ; gamma = 0.01 ; taw =  30 
# ratings = pl.read_csv("ml-32m/ratings.csv")
# train_df,test_df,user_idx_map,  movie_idx_map ,idx_to_user,idx_to_movie =  Helper_Functions.create_train_test_df(ratings)
# print(train_df.head())
# # Load movie titles
# movies_df = pl.read_csv("ml-32m/movies.csv").select(["movieId", "title"])
# movie_id_to_title = dict(zip(movies_df["movieId"].to_list(), movies_df["title"].to_list()))
# print(movie_id_to_title)
# Create idx_to_title mapping
# idx_to_title = {idx: movie_id_to_title[movie_id] for idx, movie_id in idx_to_movie.items() if movie_id in movie_id_to_title}
# print(idx_to_title[0])
# with open(os.path.join(experiment_folder, 'idx_to_title.pkl'), 'wb') as f:
#     pickle.dump(idx_to_title, f)
# # Directory where you want to save the dictionaries
# # Helper_Functions.save_idx_maps(experiment_folder,user_idx_map,  movie_idx_map ,idx_to_user,idx_to_movie)


user_idx_map,  movie_idx_map ,idx_to_user,idx_to_movie  = Helper_Functions.Load_idx_maps(f"{experiment_folder}/model_params/")
movies_factors,users_factors,user_bias,item_bias = Helper_Functions.load_model(experiment_folder)

with open(os.path.join(experiment_folder, 'idx_to_title.pkl'), 'rb') as f:
    idx_to_title = pickle.load(f)

# Step 1: Dimensionality reduction
# Using PCA (faster)
pca = PCA(n_components=2)
reduced_movie_embeddings = pca.fit_transform(movies_factors)

# OR Using t-SNE (slower but sometimes gives better clustering)
# tsne = TSNE(n_components=2, random_state=42)
# reduced_movie_embeddings = tsne.fit_transform(movies_factors)
# print()
# # # Step 2: Plot
plt.figure(figsize=(14, 10))
for i, title in list(idx_to_title.items())[1500:3000]:
    x, y = reduced_movie_embeddings[i+1500]
    plt.scatter(x, y, color='blue', s=10)  # Plot each movie point
    if i < 20 + 1500:  # Only label the first 50 movies for readability (adjust as needed)
        plt.text(x, y, title, fontsize=8)

plt.title("Movie Embeddings Visualization")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()
