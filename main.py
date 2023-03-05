import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from models import Recommender
from train import train_model
from predict import predict_ratings
from evaluate import evaluate_model

# Load user and movie data
users = pd.read_csv('data/users.csv')
movies = pd.read_csv('data/movies.csv')

user_enc = LabelEncoder()
users['user_id'] = user_enc.fit_transform(users['user_id'])
n_users = users['user_id'].nunique()

movie_enc = LabelEncoder()
movies['movie_id'] = movie_enc.fit_transform(movies['movie_id'])
n_movies = movies['movie_id'].nunique()

# Load and preprocess ratings data
ratings = pd.read_csv('data/ratings.csv')
ratings['user_id'] = user_enc.transform(ratings['user_id'])
ratings['movie_id'] = movie_enc.transform(ratings['movie_id'])

scaler = MinMaxScaler()
ratings['rating'] = scaler.fit_transform(ratings[['rating']]) # Scale ratings to between 0 and 1

# Split data into train and test sets
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2)

# Train model
model = Recommender(n_users, n_movies)
train_model(model, train_ratings)

# Predict ratings for test data
test_predictions = predict_ratings(model, test_ratings)

# Evaluate model using RMSE metric
rmse = evaluate_model(test_predictions, test_ratings)
print("RMSE: ", rmse)

# Save trained model
model.save('models/recommender.h5')
