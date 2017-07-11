import csv
import datetime      ##original code wrong
defaultencoding = 'utf-8'

def load_reviews(path, **kwargs):
    """
    Loads MoviesLens Reviews
    :param path: 
    :param kwargs: 
    :return: 
    """
    options = {
        'fieldnames': ('userid', 'movieid', 'rating',
                       'timestamp'),
        'delimiter': '\t',
    }
    options.update(kwargs)

    parse_date = lambda r,k: datetime.fromtimestamp(float(r[k]))
    parse_int =lambda r,k : int(r[k])

    with open(path, 'r' ,encoding='UTF-8') as reviews:           #
        reader = csv.DictReader(reviews, **options)
        for row in reader:
            row['user_id'] = parse_int(row, 'user_id')
            row['movieid'] = parse_int(row,'movie_id')
            row['rating'] = parse_int(row,'rating')
            row['timestamp'] = parse_date(row,'timestamp')
            yield row

import os

def relative_path(path):
    """
    returns as a path relative from this code file
    :param path:
    :return:
    """
    dirname = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(dirname, path)
    return os.path.normpath(path)


def load_movies(path, **kwargs):
    """
    Loads MovieLens movies
    """

    options = {
        'fieldnames': ('movieid', 'title', 'release',
                       'video', 'url'), 'delimiter': '|', 'restkey': 'genre',
    }
    options.update(kwargs)

    parse_int = lambda r, k: int(r[k])
    parse_date = lambda r, k:  datetime.datetime.strptime(r[k], '%d-%b-%Y') if r[k] else None       #original version is datetime.strptime,{wrong.....}

    with open(path, 'r',encoding='UTF-8') as movies:            #,encoding='UTF-8'
        reader = csv.DictReader(movies, **options)
        for row in reader:
            row['movieid'] = parse_int(row, 'movieid')
            row['release'] = parse_date(row, 'release')
            row['video'] = parse_date(row, 'video')
            yield row

from collections import defaultdict

class MovieLens(object):
    """Data structure to build our recommender model on
    """

    def __init__(self,udata,uitem):
        """Instantiate with a path to u.data and u.item"""

        self.udata = udata
        self.uitem = uitem
        self.movies = {}
        self.reviews = defaultdict(dict)
        self.load_dataset()

    def load_dataset(self):
        for movie in load_movies(self.uitem):
            self.movies[movie['movieid']] = movie

        for review in load_reviews(self.udata):
            self.reviews[review['userid']][review['movieid']] = review

#relative path varies
data = relative_path('ml-100k/u.data')
item = relative_path('ml-100k/u.item')
model = MovieLens(data, item)


