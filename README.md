
# Movie Recommender System

This is a movie recommender system built using machine learning and deep learning techniques. The system uses collaborative filtering to recommend movies to users based on their previous movie ratings and the ratings of similar users.

## Usage

To use the recommender system, simply run the `app.py` file and navigate to `http://localhost:5000/recommendations/<user_id>` in your web browser. Replace `<user_id>` with the ID of the user you want to get recommendations for.

You can also specify the number of recommendations to return by appending `?num_recs=<num>` to the URL, where `<num>` is the number of recommendations you want to return (default is 10).

## Files

The movie recommender system consists of the following files:

- `data/`: A directory containing the movie ratings data and user demographic data.
- `models.py`: A Python script containing the code for training the collaborative filtering model.
- `predict.py`: A Python script containing the code for making movie recommendations for a user.
- `app.py`: A Python script containing the Flask application for serving movie recommendations via a web API.
- `requirements.txt`: A text file containing the list of Python packages required to run the movie recommender system.

## Dependencies

The movie recommender system requires the following Python packages:

- Flask
- NumPy
- Pandas
- scikit-learn
- Surprise

To install these packages, run `pip install -r requirements.txt` in your command prompt or terminal.

## Acknowledgements

The movie ratings data used in this project is from the [MovieLens](https://grouplens.org/datasets/movielens/) dataset, which was collected by the GroupLens Research Project at the University of Minnesota. We acknowledge and thank them for making this dataset available for research purposes.
