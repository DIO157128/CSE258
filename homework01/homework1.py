import json

import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
import numpy
import random
import gzip
import math
import warnings

from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix


def assertFloat(x):  # Checks that an answer is a float
    assert type(float(x)) == float


def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float] * N


def readReview(file):
    f = open(file)
    dataset = []
    for l in f:
        dataset.append(json.loads(l))
    return dataset


def readBeer(file):
    f = open(file)
    dataset = []
    for l in f:
        if 'user/gender' in l:
            dataset.append(eval(l))
    return dataset


def Q1(datum):
    rating = []
    emark = []
    for d in datum:
        rating.append(d['rating'])
        emark.append(d['review_text'].count('!'))
    X = np.array(emark).reshape(-1, 1)
    y = np.array(rating)
    model = LinearRegression()
    model.fit(X, y)
    theta0 = model.intercept_
    theta1 = model.coef_[0]
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return theta0, theta1, mse


def Q2(datum):
    rating = []
    emark = []
    length = []
    for d in datum:
        rating.append(d['rating'])
        emark.append(d['review_text'].count('!'))
        length.append(len(d['review_text']))
    X = np.array([[l, e] for l, e in zip(length, emark)])
    y = np.array(rating)
    model = LinearRegression()
    model.fit(X, y)
    theta0 = model.intercept_
    theta1 = model.coef_[0]
    theta2 = model.coef_[1]
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return theta0, theta1, theta2, mse


def Q3(datum):
    rating = []
    emark = []
    for d in datum:
        rating.append(d['rating'])
        emark.append(d['review_text'].count('!'))
    X = np.array(emark).reshape(-1, 1)
    y = np.array(rating)
    mse = []
    for i in range(1, 6):
        model = LinearRegression()
        poly = PolynomialFeatures(degree=i)
        X_poly = poly.fit_transform(X)
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        mse.append(mean_squared_error(y, y_pred))
    return mse


def Q4(datum):
    rating = []
    emark = []
    for d in datum:
        rating.append(d['rating'])
        emark.append(d['review_text'].count('!'))
    X = np.array(emark).reshape(-1, 1)
    y = np.array(rating)
    mse = []
    half = int(len(X) / 2)
    for i in range(1, 6):
        model = LinearRegression()
        poly = PolynomialFeatures(degree=i)
        X_poly = poly.fit_transform(X)
        X_train = X_poly[:half]
        X_test = X_poly[half:]
        y_train = y[:half]
        y_test = y[half:]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse.append(mean_squared_error(y_test, y_pred))
    return mse


def Q5(datum):
    rating = []
    for d in datum:
        rating.append(d['rating'])
    y = np.array(rating)
    half = int(len(y) / 2)
    y_test = y[half:]
    mid = np.median(y_test)
    mids = [mid for i in range(len(y_test))]
    mae = mean_absolute_error(mids, y_test)
    return mae


def Q6(datum):
    gender = []
    emark = []
    for d in datum:
        gender.append(d['user/gender'])
        emark.append(d['review/text'].count('!'))
    X = np.array(emark).reshape(-1, 1)
    y = np.array(gender)
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    y_pred = log_reg.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=['Male', 'Female']).ravel()
    ber = 0.5 * (fn / (fn + tp) + fp / (fp + tn))
    return tp, tn, fp, fn, ber


def Q7(datum):
    gender = []
    emark = []
    for d in datum:
        gender.append(d['user/gender'])
        emark.append(d['review/text'].count('!'))
    X = np.array(emark).reshape(-1, 1)
    y = np.array(gender)
    log_reg = LogisticRegression(class_weight='balanced')
    log_reg.fit(X, y)
    y_pred = log_reg.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=['Male', 'Female']).ravel()
    ber = 0.5 * (fn / (fn + tp) + fp / (fp + tn))
    return tp, tn, fp, fn, ber


def Q8(datum):
    gender = []
    emark = []
    for d in datum:
        gender.append(d['user/gender'])
        emark.append(d['review/text'].count('!'))
    X = np.array(emark).reshape(-1, 1)
    y = np.array(gender)
    log_reg = LogisticRegression(class_weight='balanced')
    log_reg.fit(X, y)
    y_prob = log_reg.predict_proba(X)[:, 1]
    y_binary = np.where(y == 'Female', 1, 0)
    K_values = [1, 10, 100, 1000, 10000]
    res = []
    for k in K_values:
        top_k_indices = np.argsort(y_prob)[::-1][:k]
        top_k_labels = y_binary[top_k_indices]
        res.append(np.sum(top_k_labels) / k)
    return res


def main():
    dataset = readReview('young_adult_10000.json')
    answers = {}
    theta0, theta1, mse = Q1(dataset)
    answers['Q1'] = [theta0, theta1, mse]
    assertFloatList(answers['Q1'], 3)
    theta0, theta1, theta2, mse = Q2(dataset)
    answers['Q2'] = [theta0, theta1, theta2, mse]
    assertFloatList(answers['Q2'], 4)
    mses = Q3(dataset)
    answers['Q3'] = mses
    assertFloatList(answers['Q3'], 5)
    mses = Q4(dataset)
    answers['Q4'] = mses
    assertFloatList(answers['Q4'], 5)
    mae = Q5(dataset)
    answers['Q5'] = mae
    assertFloat(answers['Q5'])
    dataset = readBeer('beer_50000.json')
    TP, TN, FP, FN, BER = Q6(dataset)
    answers["Q6"] = [TP, TN, FP, FN, BER]
    assertFloatList(answers['Q6'], 5)
    TP, TN, FP, FN, BER = Q7(dataset)
    answers["Q7"] = [TP, TN, FP, FN, BER]
    assertFloatList(answers['Q7'], 5)
    precisionList = Q8(dataset)
    answers['Q8'] = precisionList
    assertFloatList(answers['Q8'], 5)  # List of five floats
    f = open("answers_hw1.txt", 'w')  # Write your answers to a file
    f.write(str(answers) + '\n')
    f.close()


if __name__ == '__main__':
    main()
