import os
import sys
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt


# plotting the loss and the RMSE curves
def plot_likelihood(x_axis, y_axis, experiment_folder = "./" , plot_name = "Negative Log Likelihood Curves"):
  plt.figure(figsize=(10, 6))
  plt.subplot(1,2,1)
  plt.plot(range(x_axis) , y_axis[0])
  plt.title('Training Likelihood Curve')
  plt.xlabel('Iterations')
  plt.ylabel('loss')
  plt.subplot(1,2,2)
  plt.plot(range(x_axis) , y_axis[1] , c ='r')
  plt.title('Validation Likelihood Curve')
  plt.xlabel('Iterations')
  plt.ylabel('loss')
  plot_filename = os.path.join(experiment_folder, f'_{plot_name}_experiment_plot.pdf')    
  print(f"Plot saved as {plot_name} Curves")
  plt.savefig(plot_filename)
  plt.show()
  plt.close() 

# Movie degrees
def plot_rmse(x_axis,y_axis,experiment_folder = './',plot_name = "RMSE Curves"):
  plt.figure(figsize=(10, 6))
  plt.plot(range(x_axis) , y_axis[0])
  plt.plot(range(x_axis) , y_axis[1] , c ='r')
  plt.title('Training & Validation RMSE Curves')
  plt.xlabel('Iterations')
  plt.ylabel('RMSE')
  plot_filename = os.path.join(experiment_folder, f'_{plot_name}_experiment_plot.pdf')    
  print(f"Plot saved as {plot_name}  Curves")
  plt.savefig(plot_filename)
  plt.show()
  plt.close() 

def setup_logging(experiment_folder):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create a file handler to log to a file in the experiment folder
    log_filename = os.path.join(experiment_folder, 'experiment_log.txt')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Create a stream handler to print to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def setup_experiment_folder(experiment_name):
    # Create a directory for the experiment if it doesn't exist
    experiment_folder = f"./Experiments/{experiment_name}/"
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    return experiment_folder


def create_fake_user(list_of_favourite_movies,item_bias,movies_factors,K_factors,lambda_reg,gamma,taw):
    movie_indices , rating = list_of_favourite_movies

    movie_bias = item_bias[movie_indices]
    movie_factor = movies_factors[movie_indices]
    asim = np.zeros((1 ,K_factors))
    b_asim = lambda_reg *( rating -  movie_bias )/(lambda_reg +gamma)

    for _ in range(5):
        # update Asim

        sum_VnVnT = lambda_reg * np.outer(movie_factor,movie_factor)
        reg_term = taw*np.eye(K_factors)
        sum_rmn_Vn = lambda_reg * (movie_factor * (rating - b_asim - movie_bias  ))
        asim = np.linalg.solve(sum_VnVnT + reg_term, sum_rmn_Vn)
        # update b_asim again
        rhat = asim@movie_factor
        b_asim = lambda_reg *(rating - rhat- movie_bias )/(lambda_reg +gamma)
    return asim 


def save_model(experiment_folder,movies_factors,users_factors,user_bias,item_bias):
  if not os.path.exists(f'{experiment_folder}model_params'):
        os.makedirs(f'{experiment_folder}model_params')
  np.save(f'{experiment_folder}model_params/user_bias.npy', user_bias,allow_pickle=True)
  np.save(f'{experiment_folder}model_params/item_bias.npy', item_bias,allow_pickle=True)
  np.save(f'{experiment_folder}model_params/users_factors.npy', users_factors,allow_pickle=True)
  np.save(f'{experiment_folder}model_params/movies_factors.npy', movies_factors,allow_pickle=True)

def load_model(experiment_folder):
    #   for testing the model
    user_bias      = np.load(f'{experiment_folder}model_params/user_bias.npy',allow_pickle=True)
    item_bias      = np.load(f'{experiment_folder}model_params/item_bias.npy',allow_pickle=True)
    users_factors  = np.load(f'{experiment_folder}model_params/users_factors.npy',allow_pickle=True)
    movies_factors = np.load(f'{experiment_folder}model_params/movies_factors.npy',allow_pickle=True)
    return movies_factors,users_factors,user_bias,item_bias

def Load_idx_maps(load_dir):
    with open(os.path.join(load_dir, 'user_idx_map.pkl'), 'rb') as f:
        user_idx_map = pickle.load(f)

    with open(os.path.join(load_dir, 'movie_idx_map.pkl'), 'rb') as f:
        movie_idx_map = pickle.load(f)

    with open(os.path.join(load_dir, 'idx_to_user.pkl'), 'rb') as f:
        idx_to_user = pickle.load(f)

    with open(os.path.join(load_dir, 'idx_to_movie.pkl'), 'rb') as f:
        idx_to_movie = pickle.load(f)

    with open(os.path.join(load_dir, 'genre_to_idx.pkl'), 'rb') as f:
        genre_to_idx = pickle.load(f)

    with open(os.path.join(load_dir, 'genre_to_movies.pkl'), 'rb') as f:
        genre_to_movies = pickle.load(f)
    return user_idx_map,  movie_idx_map ,idx_to_user,idx_to_movie,genre_to_idx,genre_to_movies


def Load_training_data(experiment_folder):
    users_train        = np.load(f'./{experiment_folder}/users_train.npy',allow_pickle=True)
    movies_train       = np.load(f'./{experiment_folder}/movies_train.npy',allow_pickle=True)
    movies_train_idxes = np.load(f'./{experiment_folder}/movies_train_idxes.npy',allow_pickle=True)
    users_train_idxes  = np.load(f'./{experiment_folder}/users_train_idxes.npy', allow_pickle=True)
    movies_genres_array= np.load(f'./{experiment_folder}/movies_genres_array.npy', allow_pickle=True)
    # genre_to_movies    = np.load(f'./{experiment_folder}/genre_to_movies.npy', allow_pickle=True)
    return users_train,movies_train,movies_train_idxes,users_train_idxes,movies_genres_array

def Load_test_data(experiment_folder):
    users_test         = np.load(f'./{experiment_folder}/users_test.npy' ,allow_pickle=True)
    movies_test        = np.load(f'./{experiment_folder}/movies_test.npy', allow_pickle=True)
    users_test_idxes   = np.load(f'./{experiment_folder}/users_test_idxes.npy', allow_pickle=True)
    movies_test_idxes  = np.load(f'./{experiment_folder}/movies_test_idxes.npy', allow_pickle=True)
    return users_test,movies_test,users_test_idxes,movies_test_idxes 

