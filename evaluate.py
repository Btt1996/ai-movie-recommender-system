import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load user and movie data
users = pd.read_csv('data/users.csv')
movies = pd.read_csv('data/movies.csv')

user_enc = LabelEncoder()
users['user_id'] = user_enc.fit_transform(users['user_id'])
n_users = users['user_id'].nunique()

movie_enc = LabelEncoder()
movies['movie_id'] = movie_enc.fit_transform(movies['movie_id'])
n_movies = movies['movie_id'].nunique()

# Load and preprocess test data
test_ratings = pd.read_csv('data/test_ratings.csv')
test_ratings['user_id'] = user_enc.transform(test_ratings['user_id'])
test_ratings['movie_id'] = movie_enc.transform(test_ratings['movie_id'])

scaler = MinMaxScaler()
test_ratings['rating'] = scaler.fit_transform(test_ratings[['rating']]) # Scale ratings to between 0 and 1

# Load trained model
model = tf.keras.models.load_model('models/recommender.h5')

# Predict ratings for test data
test_predictions = model.predict([test_ratings['user_id'], test_ratings['movie_id']])

# Evaluate model using RMSE metric
mse = np.mean(np.square(test_predictions - test_ratings['rating']))
rmse = np.sqrt(mse)
print("RMSE: ", rmse)
