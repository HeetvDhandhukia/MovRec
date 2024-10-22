import streamlit as st
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load the datasets
movies = pd.read_csv(r"C:\Users\heetd\OneDrive\Desktop\MovRec\movies.csv")
ratings = pd.read_csv(r"C:\Users\heetd\OneDrive\Desktop\MovRec\ratings.csv")

# Data preprocessing
final_dataset = ratings.pivot(index="movieId", columns="userId", values="rating")
final_dataset.fillna(0, inplace=True)

no_user_voted = ratings.groupby("movieId")['rating'].agg('count')
no_movies_voted = ratings.groupby("userId")['rating'].agg('count')

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

# Convert to CSR matrix
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Train the KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Function to get movie recommendations
def get_recommendation(movie_name):
    movie_list = movies[movies['title'].str.contains(movie_name, case=False)]
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distance, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=11)
        rec_movies_indices = sorted(list(zip(indices.squeeze().tolist(), distance.squeeze().tolist())), key=lambda x: x[1])[:0: -1]
        recommended_movies = []
        for val in rec_movies_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommended_movies.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        df = pd.DataFrame(recommended_movies, index=range(1, 11))
        return df
    else:
        return "Movie not found..."

# Streamlit UI
st.title("Movie Recommendation System")

# Input movie name
movie_name = st.text_input("Enter a movie name:")

# If movie name is provided, display recommendations
if movie_name:
    recommendations = get_recommendation(movie_name)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write("Recommended Movies:")
        st.dataframe(recommendations)
