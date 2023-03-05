import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load and preprocess user and movie data
users = pd.read_csv('data/users.csv')
movies = pd.read_csv('data/movies.csv')

user_enc = LabelEncoder()
users['user_id'] = user_enc.fit_transform(users['user_id'])
n_users = users['user_id'].nunique()

movie_enc = LabelEncoder()
movies['movie_id'] = movie_enc.fit_transform(movies['movie_id'])
n_movies = movies['movie_id'].nunique()

scaler = MinMaxScaler()
ratings = pd.read_csv('data/ratings.csv')
ratings['user_id'] = user_enc.transform(ratings['user_id'])
ratings['movie_id'] = movie_enc.transform(ratings['movie_id'])
ratings['rating'] = scaler.fit_transform(ratings[['rating']]) # Scale ratings to between 0 and 1

# Split data into training and validation sets
train, val = train_test_split(ratings, test_size=0.2, random_state=42)

# Define model architecture
def recommender_model(n_users, n_movies, embedding_size=50):
    user_input = Input(shape=[1])
    movie_input = Input(shape=[1])

    user_emb = Embedding(n_users, embedding_size)(user_input)
    movie_emb = Embedding(n_movies, embedding_size)(movie_input)

    user_emb = Flatten()(user_emb)
    movie_emb = Flatten()(movie_emb)

    dot_product = Dot(axes=1)([user_emb, movie_emb])

    user_bias = Embedding(n_users, 1)(user_input)
    user_bias = Flatten()(user_bias)

    movie_bias = Embedding(n_movies, 1)(movie_input)
    movie_bias = Flatten()(movie_bias)

    add_bias = Add()([dot_product, user_bias, movie_bias])

    output = Dense(1, activation='sigmoid')(add_bias)

    model = Model([user_input, movie_input], output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train model
model = recommender_model(n_users, n_movies, embedding_size=50)
history = model.fit([train['user_id'], train['movie_id']], train['rating'], 
                    validation_data=([val['user_id'], val['movie_id']], val['rating']),
                    epochs=10, batch_size=256)

# Save model
model.save('models/recommender.h5')
