import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load movie ratings data
ratings = pd.read_csv('data/ratings.csv')

# Preprocess ratings data
scaler = MinMaxScaler()
ratings['rating'] = scaler.fit_transform(ratings[['rating']])
user_enc = LabelEncoder()
ratings['user_id'] = user_enc.fit_transform(ratings['user_id'])
n_users = ratings['user_id'].nunique()
movie_enc = LabelEncoder()
ratings['movie_id'] = movie_enc.fit_transform(ratings['movie_id'])
n_movies = ratings['movie_id'].nunique()

# Split ratings data into train and test sets
X = ratings[['user_id', 'movie_id']].values
y = ratings['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model architecture
user_input = Input(shape=[1])
user_emb = Embedding(n_users, 10)(user_input)
user_emb = Flatten()(user_emb)
movie_input = Input(shape=[1])
movie_emb = Embedding(n_movies, 10)(movie_input)
movie_emb = Flatten()(movie_emb)
concat = concatenate([user_emb, movie_emb])
dense = Dense(128, activation='relu')(concat)
dropout = Dropout(0.5)(dense)
output = Dense(1, activation='sigmoid')(dropout)
model = Model(inputs=[user_input, movie_input], outputs=output)

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit([X_train[:,0], X_train[:,1]], y_train, batch_size=64, epochs=10, verbose=1, validation_data=([X_test[:,0], X_test[:,1]], y_test))

# Save trained model
model.save('models/recommender.h5')
