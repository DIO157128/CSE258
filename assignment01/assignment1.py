import gzip
import random
from collections import defaultdict
from surprise import SVD, Dataset, Reader, SVDpp
import numpy as np


# ----------------------------- Utility Functions ----------------------------- #

def read_csv(path, has_header=False):
    """Read a gzipped CSV file and yield rows."""
    with open(path, 'rt') as f:
        if has_header:
            f.readline()  # Skip header
        for line in f:
            yield line.strip().split(',')


def jaccard_similarity(set1, set2):
    """Calculate the Jaccard similarity between two sets."""
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    if union_size == 0:
        return 0
    return intersection_size / union_size

# ----------------------------- Prediction Functions ----------------------------- #

def generate_popularity_baseline(users_per_book, total_reads, threshold=0.7):
    """Generate a popularity-based recommendation baseline."""
    most_popular = sorted(
        [(len(users), book) for book, users in users_per_book.items()],
        reverse=True
    )
    popular_books = set()
    count = 0
    for num_users, book in most_popular:
        count += num_users
        popular_books.add(book)
        if count > total_reads * threshold:
            break
    return popular_books


def predict_read(train_file, pairs_file, output_file, jac_threshold=0.1):
    """Generate predictions for 'read' task using popularity and Jaccard similarity."""
    # Initialize data structures
    ratings_per_user = defaultdict(list)
    books_per_user = defaultdict(set)
    users_per_book = defaultdict(set)
    total_read = 0

    # Load training data
    for user, book, rating in read_csv(train_file):
        rating = int(rating)
        ratings_per_user[user].append((book, rating))
        books_per_user[user].add(book)
        users_per_book[book].add(user)
        total_read += 1

    # Generate popularity baseline
    popular_books = generate_popularity_baseline(users_per_book, total_read)

    # Generate predictions
    with open(output_file, 'w') as predictions:
        for line in open(pairs_file):
            if line.startswith("userID"):
                predictions.write(line)
                continue

            user, book = line.strip().split(',')
            max_jaccard = 0

            # Find the most similar book
            for user_book in books_per_user[user]:
                similarity = jaccard_similarity(users_per_book[book], users_per_book[user_book])
                max_jaccard = max(max_jaccard, similarity)

            # Predict based on popularity or Jaccard similarity
            if book in popular_books or max_jaccard > jac_threshold:
                predictions.write(f"{user},{book},1\n")
            else:
                predictions.write(f"{user},{book},0\n")


def predict_rating(train_file, pairs_file, output_file):
    """Generate predictions for 'rating' task using SVD++."""
    # Load data using Surprise
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_file(train_file, reader=reader)
    trainset = data.build_full_trainset()

    # Train SVD++ model
    model = SVDpp(reg_all=0.2)
    model.fit(trainset)

    # Generate predictions
    with open(output_file, 'w') as predictions:
        for line in open(pairs_file):
            if line.startswith("userID"):
                predictions.write(line)
                continue

            user, book = line.strip().split(',')
            rating_prediction = model.predict(user, book)
            predictions.write(f"{user},{book},{rating_prediction.est}\n")


# ----------------------------- Main Script ----------------------------- #

if __name__ == '__main__':
    # File paths
    train_file = "train_Interactions.csv"
    pairs_read_file = "pairs_Read.csv"
    pairs_rating_file = "pairs_Rating.csv"
    predictions_read_file = "predictions_Read.csv"
    predictions_rating_file = "predictions_Rating.csv"

    # Generate predictions for both tasks
    predict_read(train_file, pairs_read_file, predictions_read_file)
    predict_rating(train_file, pairs_rating_file, predictions_rating_file)
