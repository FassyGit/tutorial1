import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tag_headers = ['user_id' ,'movie_id', 'tag' ,'timestamp']
tags = pd.read_table('data/tags.dat', sep='::',header = None, names = tag_headers)

rating_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('data/ratings.dat',sep='::',header=None,names =rating_headers)

movie_headers = ['movie_id', 'title', 'genres']
movies = pd.read_table('data/movies.dat',sep='::', header=None, names=movie_headers)
movie_titles = movies.title.tolist()
movies.head()

ratings.head()
tags.head()
df = movies.join(ratings, on=['movie_id'], rsuffix='_r').join(tags, on=['movie_id'], rsuffix='_t')
del df['movie_id_r']
del df['user_id_t']
del df['movie_id_t']
del df['timestamp_t']
rp = df.pivot_table(cols=['movie_id'],rows=['user_id'],values='rating')
rp.head()
rp = rp.fillna(0); # Replace NaN
rp.head()
Q = rp.values
Q.shape
W = Q>0.5
W[W == True] = 1
W[W == False] = 0
# To be consistent with our Q matrix
W = W.astype(np.float64, copy=False)
