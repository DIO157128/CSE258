import re

import numpy
import urllib

import numpy as np
import pandas as pd
import scipy.optimize
import random

import tqdm
from sklearn import linear_model
import gzip
from collections import defaultdict
import warnings

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, jaccard_score

warnings.filterwarnings("ignore")


def assertFloat(x):
    assert type(float(x)) == float


def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float] * N


def readBank(file):
    f = open(file, 'r')
    while not '@data' in f.readline():
        pass

    dataset = []
    for l in f:
        if '?' in l:  # Missing entry
            continue
        l = l.split(',')
        values = [1] + [float(x) for x in l]
        values[-1] = values[-1] > 0  # Convert to bool
        dataset.append(values)

    return dataset


def readBook(file):
    f = gzip.open(file)
    dataset = []
    for l in f:
        dataset.append(eval(l))
    return dataset


def accuracy(predictions, y):
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    return (tp + tn) / (tp + tn + fp + fn)


def BER(predictions, y):
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    return 0.5 * (fn / (fn + tp) + fp / (fp + tn))


def Q1(dataset):
    X = [d[:-1] for d in dataset]
    y = [d[-1] for d in dataset]
    model = linear_model.LogisticRegression(C=1)
    model.fit(X, y)
    y_pred = model.predict(X)
    return accuracy(y_pred, y), BER(y_pred, y)


def Q2(dataset):
    X = [d[:-1] for d in dataset]
    y = [d[-1] for d in dataset]
    model = linear_model.LogisticRegression(C=1, class_weight='balanced')
    model.fit(X, y)
    y_pred = model.predict(X)
    return accuracy(y_pred, y), BER(y_pred, y)


def Q3(dataset):
    random.seed(3)
    random.shuffle(dataset)
    X = [d[:-1] for d in dataset]
    y = [d[-1] for d in dataset]
    Xtrain, Xvalid, Xtest = X[:len(X) // 2], X[len(X) // 2:(3 * len(X)) // 4], X[(3 * len(X)) // 4:]
    ytrain, yvalid, ytest = y[:len(X) // 2], y[len(X) // 2:(3 * len(X)) // 4], y[(3 * len(X)) // 4:]
    model = linear_model.LogisticRegression(C=1, class_weight='balanced')
    model.fit(Xtrain, ytrain)
    y_pred_train = model.predict(Xtrain)
    y_pred_valid = model.predict(Xvalid)
    y_pred_test = model.predict(Xtest)
    return BER(y_pred_train, ytrain), BER(y_pred_valid, yvalid), BER(y_pred_test, ytest)


def Q4(dataset):
    random.seed(3)
    random.shuffle(dataset)
    X = [d[:-1] for d in dataset]
    y = [d[-1] for d in dataset]
    Xtrain, Xvalid, Xtest = X[:len(X) // 2], X[len(X) // 2:(3 * len(X)) // 4], X[(3 * len(X)) // 4:]
    ytrain, yvalid, ytest = y[:len(X) // 2], y[len(X) // 2:(3 * len(X)) // 4], y[(3 * len(X)) // 4:]
    Cs = [1e-04, 1e-03, 1e-02, 1e-01, 1e+00, 1e+01, 1e+02, 1e+03, 1e+04]
    bers = []
    for C in Cs:
        model = linear_model.LogisticRegression(C=C, class_weight='balanced')
        model.fit(Xtrain, ytrain)
        y_pred = model.predict(Xvalid)
        bers.append(BER(y_pred, yvalid))
    return bers


def Q5(dataset):
    random.seed(3)
    random.shuffle(dataset)
    X = [d[:-1] for d in dataset]
    y = [d[-1] for d in dataset]
    Xtrain, Xvalid, Xtest = X[:len(X) // 2], X[len(X) // 2:(3 * len(X)) // 4], X[(3 * len(X)) // 4:]
    ytrain, yvalid, ytest = y[:len(X) // 2], y[len(X) // 2:(3 * len(X)) // 4], y[(3 * len(X)) // 4:]
    Cs = [1e-04, 1e-03, 1e-02, 1e-01, 1e+00, 1e+01, 1e+02, 1e+03, 1e+04]
    bers = []
    for C in Cs:
        model = linear_model.LogisticRegression(C=C, class_weight='balanced')
        model.fit(Xtrain, ytrain)
        y_pred = model.predict(Xvalid)
        bers.append(BER(y_pred, yvalid))
    min_index = np.argmin(bers)
    C = Cs[min_index]
    ber = bers[min_index]
    return C, ber


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


def Q6(dataset):
    dataTrain = dataset[:9000]
    usersPerItem = defaultdict(set)  # Maps an item to the users who rated it
    itemsPerUser = defaultdict(set)  # Maps a user to the items that they rated
    ratingDict = {}  # To retrieve a rating for a specific user/item pair

    for d in dataTrain:
        user, item = d['user_id'], d['book_id']
        usersPerItem[item].add(user)
        itemsPerUser[user].add(item)
        ratingDict[(user, item)] = d['rating']
    similarities = []
    users_0 = usersPerItem[dataTrain[0]['book_id']]
    for i in usersPerItem:
        if i == dataTrain[0]['book_id']:
            continue
        users_cur = usersPerItem[i]
        sim = Jaccard(users_0, users_cur)
        similarities.append((sim, i))
    similarities = sorted(similarities, key=lambda x: x[0], reverse=True)[:10]
    return similarities


def MSE(predictions, labels):
    differences = [(x - y) ** 2 for x, y in zip(predictions, labels)]
    return sum(differences) / len(differences)


def Q7(dataset):
    dataTrain = dataset[:9000]
    dataTest = dataset[9000:]
    usersPerItem = defaultdict(set)
    itemsPerUser = defaultdict(set)
    reviewsPerUser = defaultdict(list)
    reviewsPerItem = defaultdict(list)
    ratingDict = {}
    for d in dataTrain:
        user, item = d['user_id'], d['book_id']
        usersPerItem[item].add(user)
        itemsPerUser[user].add(item)
        ratingDict[(user, item)] = d['rating']
        reviewsPerUser[user].append(d)
        reviewsPerItem[item].append(d)
    userAverages = {}
    itemAverages = {}
    for u in itemsPerUser:
        rs = [ratingDict[(u, i)] for i in itemsPerUser[u]]
        userAverages[u] = sum(rs) / len(rs)
    for i in usersPerItem:
        rs = [ratingDict[(u, i)] for u in usersPerItem[i]]
        itemAverages[i] = sum(rs) / len(rs)
    ratingMean = sum([d['rating'] for d in dataset]) / len(dataset)

    def predictRating(user, item):
        ratings = []
        similarities = []
        if item not in itemAverages.keys():
            return ratingMean
        for d in reviewsPerUser[user]:
            cur_item = d['book_id']
            if cur_item == item: continue
            ratings.append(d['rating'] - itemAverages[cur_item])
            similarities.append(Jaccard(usersPerItem[item], usersPerItem[cur_item]))
        if (sum(similarities) > 0):
            weightedRatings = [(x * y) for x, y in zip(ratings, similarities)]
            return itemAverages[item] + sum(weightedRatings) / sum(similarities)
        else:
            return ratingMean

    predictions, labels = [], []
    for d in tqdm.tqdm(dataTest):
        user_id, book_id, rating = d['user_id'], d['book_id'], d['rating']
        predictions.append(predictRating(user_id, book_id))
        labels.append(rating)
    return MSE(predictions, labels)


def Q8(dataset):
    dataTrain = dataset[:9000]
    dataTest = dataset[9000:]
    usersPerItem = defaultdict(set)
    itemsPerUser = defaultdict(set)
    reviewsPerUser = defaultdict(list)
    reviewsPerItem = defaultdict(list)
    ratingDict = {}
    for d in dataTrain:
        user, item = d['user_id'], d['book_id']
        usersPerItem[item].add(user)
        itemsPerUser[user].add(item)
        ratingDict[(user, item)] = d['rating']
        reviewsPerUser[user].append(d)
        reviewsPerItem[item].append(d)
    userAverages = {}
    itemAverages = {}
    for u in itemsPerUser:
        rs = [ratingDict[(u, i)] for i in itemsPerUser[u]]
        userAverages[u] = sum(rs) / len(rs)
    for i in usersPerItem:
        rs = [ratingDict[(u, i)] for u in usersPerItem[i]]
        itemAverages[i] = sum(rs) / len(rs)
    ratingMean = sum([d['rating'] for d in dataset]) / len(dataset)

    def predictRating(user, item):
        ratings = []
        similarities = []
        if user not in userAverages.keys():
            return ratingMean
        for d in reviewsPerItem[item]:
            cur_user = d['user_id']
            if cur_user == user: continue
            ratings.append(d['rating'] - userAverages[cur_user])
            similarities.append(Jaccard(itemsPerUser[user], itemsPerUser[cur_user]))
        if (sum(similarities) > 0):
            weightedRatings = [(x * y) for x, y in zip(ratings, similarities)]
            return userAverages[user] + sum(weightedRatings) / sum(similarities)
        else:
            return ratingMean

    predictions, labels = [], []
    for d in tqdm.tqdm(dataTest):
        user_id, book_id, rating = d['user_id'], d['book_id'], d['rating']
        predictions.append(predictRating(user_id, book_id))
        labels.append(rating)
    return MSE(predictions, labels)


def main():
    answers = {}
    dataset = readBank("5year.arff")
    acc1, ber1 = Q1(dataset)
    answers['Q1'] = [acc1, ber1]  # Accuracy and balanced error rate
    assertFloatList(answers['Q1'], 2)
    acc2, ber2 = Q2(dataset)
    answers['Q2'] = [acc2, ber2]
    assertFloatList(answers['Q2'], 2)
    berTrain, berValid, berTest = Q3(dataset)
    answers['Q3'] = [berTrain, berValid, berTest]
    assertFloatList(answers['Q3'], 3)
    berList = Q4(dataset)
    answers['Q4'] = berList
    assertFloatList(answers['Q4'], 9)
    bestC, ber5 = Q5(dataset)
    answers['Q5'] = [bestC, ber5]
    assertFloatList(answers['Q5'], 2)
    dataset = readBook('young_adult_10000.json.gz')
    answers['Q6'] = Q6(dataset)
    assert len(answers['Q6']) == 10
    assertFloatList([x[0] for x in answers['Q6']], 10)
    answers['Q7'] = Q7(dataset)
    assertFloat(answers['Q7'])
    answers['Q8'] = Q8(dataset)
    assertFloat(answers['Q8'])
    f = open("answers_hw2.txt", 'w')
    f.write(str(answers) + '\n')
    f.close()


if __name__ == '__main__':
    main()
