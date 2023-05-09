import pandas as pd
import numpy
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from os.path import exists

MOVIES_AMOUNT = 10
AMOUNT_FOR_ROW = 3
LIKED_MOVIES_AMOUNT = 3


def get_director(crew):
    """
    Gets director from a crew list.
    :param crew: crew list
    :type: list
    :return: director name, if exists
    :rtype: string
    """
    for member in crew:
        if member["job"] == "Director":
            return member["name"]

    return numpy.nan


def get_list(data_list, feature):
    """
    get names list from a movie data frame feature.
    :param data_list: data list of a feature (contains name + job + gender, for example)
    :type data_list: list
    :param feature: feature name
    :type feature: string
    :return: list containing only names
    :rtype: list
    """
    if isinstance(data_list, list):
        names = [i["name"] for i in data_list]

        if len(names) > AMOUNT_FOR_ROW and feature != "genres":
            names = names[:AMOUNT_FOR_ROW]  # Leaves only most important

        return names

    return []


def create_soup(column):
    """
    Creates soup from all necessary movie information.
    :param column: data column of specific movie
    :type column: pandas.core.series.Series
    :return: soup
    :rtype: string
    """
    return ' '.join(column['keywords']) + ' ' + ' '.join(column['cast']) \
           + ' ' + column['director'] + ' ' + ' '.join(column['genres'])


def clean_data(row):
    """
    lowers data letters and removes whitespaces.
    :param row: row of data in dataframe
    :type row: list or str
    :return: cleaned data
    :rtype: list or str, respectively
    """
    if isinstance(row, list):
        return [str.lower(i.replace(" ", "")) for i in row]
    else:
        if isinstance(row, str):
            return str.lower(row.replace(" ", ""))
        else:
            return ""


def sort_movies(recommendations, movies):
    """
    Sorts movies from most similar to less.
    :param recommendations: recommended movies
    :type recommendations: list
    :param movies: movies liked by the user
    :type movies: list
    :return: movies similar to those like by the user, ordered
    :rtype: list
    """
    # Remove already-watched movies from list
    recommendations = [title for title in recommendations if title.lower() not in movies]
    # Sorts movies by most similarity to the 3 selected movies
    commons = Counter(recommendations).most_common()

    return [movie for movie, count in commons][:MOVIES_AMOUNT]  # 10 most common movies


def get_similar_movies(movies_df, cosine_sim):
    """
    Gets movies similar to the movies chosen by the user.
    :param movies_df: movies data frame.
    :type movies_df: pandas.core.frame.DataFrame
    :param cosine_sim: matrix containing all movies cosine similarities
    :type cosine_sim: numpy.ndarray
    :return: similar movies found
    :rtype: list
    """
    similar_movies = []
    for i in range(LIKED_MOVIES_AMOUNT):
        # Gets 10 most similar movies
        sim_scores = list(enumerate(cosine_sim[i]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:MOVIES_AMOUNT + 1]
        # Adds movies to similar movies list
        movies_indices = [index[0] for index in sim_scores]
        similar_movies += movies_df["title"].iloc[movies_indices].values.tolist()

    return similar_movies


def merge_information(credits_df, movies_df):
    """
    Merges movies data frame and credits data frame.
    :param credits_df: data frame containing cast and crew for each movie
    :type credits_df: pandas.core.frame.DataFrame
    :param movies_df: data frame containing keywords and genres for each movie
    :type movies_df: pandas.core.frame.DataFrame
    :return: full data frame created from both data frames
    :rtype: pandas.core.frame.DataFrame
    """
    credits_df.columns = ['id', 'title', 'cast', 'crew']
    cols_to_include = ['id', 'cast', 'crew']
    movies_df = movies_df.merge(credits_df[cols_to_include], on="id")

    return movies_df


def prepare_info(movies_df):
    """
    Arranges data, adds director row and prepares data for soup creation
    :param movies_df: movies data frame.
    :type movies_df: pandas.core.frame.DataFrame
    :return: None
    """
    # Turns list-like and dict-like strings into lists and dicts
    features = ["cast", "crew", "keywords", "genres"]
    for feature in features:
        movies_df[feature] = movies_df[feature].apply(literal_eval)
    movies_df["director"] = movies_df["crew"].apply(get_director)

    features = ["cast", "keywords", "genres"]
    for feature in features:
        # Leaves only the important information in the features
        movies_df[feature] = movies_df[feature].apply(lambda x: get_list(x, feature="genres"))

    features = ['cast', 'keywords', 'director', 'genres']
    for feature in features:
        # Removes whitespaces and lowers strings
        movies_df[feature] = movies_df[feature].apply(clean_data)


def get_movies_df():
    """
    Gets movies data frame from data files.
    :return: movies data frame.
    :rtype: pandas.core.frame.DataFrame
    """
    movies_df = pd.DataFrame
    if exists("full_movies_info.csv"):  # Manipulations on dataset were done once before
        movies_df = pd.read_csv("full_movies_info.csv")
    else:
        try:
            credits_df = pd.read_csv("credits_db.csv")
            movies_df = pd.read_csv("movies_db.csv")
        except FileNotFoundError as err:
            print(err)
            exit(input())
        else:
            movies_df = merge_information(credits_df, movies_df)
            prepare_info(movies_df)  # Prepares info for soup creation
            movies_df["soup"] = movies_df.apply(create_soup, axis=1)
            # Drops all unnecessary columns
            movies_df.drop(movies_df.columns.difference(["soup", "title"]), inplace=True, axis=1)

            with open("full_movies_info.csv", 'w', encoding="utf-8") as f:
                f.write(movies_df.to_csv())
    return movies_df


def get_cosine_similarity(soups, indices, liked_movies):
    """
    Gets cosine similarity of all movies.
    :param liked_movies: movies liked by the user
    :type liked_movies: list
    :param indices: mapping of each movie title with its index
    :type indices: pandas.core.series.Series
    :param soups: all movies soups. Soup is a mix of important information.
    :type soups: pandas.core.series.Series
    :return: matrix of all cosine similarities.
    :rtype: numpy.ndarray
    """
    vocabulary = set([word for soup in soups for word in soup.split()])  # All words used in all movies soups
    #  Get words count for all movies soups
    count_vectorizer = CountVectorizer(stop_words="english", vocabulary=vocabulary)
    movies_matrix = count_vectorizer.fit_transform(soups)

    liked_movies_soups = soups.iloc[indices.loc[liked_movies]]

    liked_movies_matrix = count_vectorizer.fit_transform(liked_movies_soups)
    # Gets similarity between movies
    cosine_sim = cosine_similarity(liked_movies_matrix, movies_matrix)

    return cosine_sim


def get_liked_movies(indices):
    """
    Gets movies liked by the user.
    :return: movies liked by the user.
    :rtype: list
    """
    print("Enter names of 3 movies you like:")
    print("---------------------------------------")
    liked_movies = []
    for i in range(LIKED_MOVIES_AMOUNT):
        movie_found = False
        while not movie_found:
            movie = input(f"Enter {i + 1}th name: ").lower()
            if movie not in indices.keys():
                print("Movie wasn't found, try another one.")
            else:
                movie_found = True
                liked_movies.append(movie)

    return liked_movies


def get_recommendations(movies_df, liked_movies, cosine_sim):
    """
    Gets recommended movies for the user based on liked movies.
    :param movies_df: movies data frame.
    :type movies_df: pandas.core.frame.DataFrame
    :param liked_movies: movies liked by the user.
    :type liked_movies: list
    :param cosine_sim: contains the cosine similarities between all movies.
    :type cosine_sim: numpy.ndarray
    :return: recommended movies for the user to watch.
    :rtype: list
    """
    similar_movies = get_similar_movies(movies_df, cosine_sim)
    recommended_movies = sort_movies(similar_movies, liked_movies)

    return recommended_movies


def main():
    movies_df = get_movies_df()
    # Enables getting movie's index by title, and vice versa
    indices = pd.Series(movies_df.index, index=movies_df['title'].apply(str.lower)).drop_duplicates()

    liked_movies = get_liked_movies(indices)
    cosine_sim = get_cosine_similarity(movies_df["soup"], indices, liked_movies)

    recommended_movies = get_recommendations(movies_df, liked_movies, cosine_sim)
    print(f"\nMovies found! Here are the most recommended for you:")
    print("------------------------------------------------------------------")
    for i in range(len(recommended_movies)):
        print(f"{i + 1}. {recommended_movies[i]}")

    input()


if __name__ == "__main__":
    main()
