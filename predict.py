import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load saved model
model = load_model('models/recommender.h5')

# Load user and movie data
users = pd.read_csv('data/users.csv')
movies = pd.read_csv('data/movies.csv')

# Preprocess user and movie data
user_enc = LabelEncoder()
users['user_id'] = user_enc.fit_transform(users['user_id'])
n_users = users['user_id'].nunique()
movie_enc = LabelEncoder()
movies['movie_id'] = movie_enc.fit_transform(movies['movie_id'])
n_movies = movies['movie_id'].nunique()

# Generate movie recommendations for a user
def predict_movies(user_id, n_recommendations=10):
    # Create user and movie arrays
    user_arr = np.array([user_id])
    movie_arr = np.array(range(n_movies))
    # Generate rating predictions for all movies
    predictions = model.predict([user_arr, movie_arr])
    # Convert predictions to movie ratings
    ratings = predictions.reshape(-1)
    # Sort ratings and get top recommendations
    top_ratings = ratings.argsort()[::-1][:n_recommendations]
    top_movies = [movies.loc[movies['movie_id'] == movie_enc.inverse_transform([i])[0], 'title'].values[0] for i in top_ratings]
    return top_movies

# Example usage
if __name__ == '__main__':
    user_id = 100
    recommendations = predict_movies(user_id, n_recommendations=10)
    print(f'Top recommendations for user {user_id}:')
    for i, movie in enumerate(recommendations):
        print(f'{i+1}. {movie}')
