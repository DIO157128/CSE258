import json
import gzip
import math
import numpy
from collections import defaultdict

import numpy as np
import sklearn
import tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression
import random
import statistics
from sklearn.metrics import confusion_matrix, jaccard_score


def assertFloat(x):
    assert type(float(x)) == float


def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float] * N


answers = {}


def readDataset():
    z = gzip.open("./steam.json.gz")
    dataset = []
    for l in z:
        d = eval(l)
        dataset.append(d)
    z.close()
    return dataset


def MSE(predictions, labels):
    differences = [(x - y) ** 2 for x, y in zip(predictions, labels)]
    return sum(differences) / len(differences)


def BER(predictions, y):
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    return 0.5 * (fn / (fn + tp) + fp / (fp + tn))


def Q1(dataset):
    review_length = []
    time_played = []
    for d in dataset:
        review_length.append(len(d['text']))
        time_played.append(d['hours'])
    X = np.array(review_length).reshape(-1, 1)
    y = np.array(time_played)
    model = LinearRegression()
    model.fit(X, y)
    theta0 = model.intercept_
    theta1 = model.coef_[0]
    y_pred = model.predict(X)
    mse = MSE(y, y_pred)
    return theta1, mse


def Q2(dataset):
    dataTrain = dataset[:int(len(dataset) * 0.8)]
    dataTest = dataset[int(len(dataset) * 0.8):]
    review_length_train = []
    time_played_train = []
    review_length_test = []
    time_played_test = []
    for d in dataTrain:
        review_length_train.append(len(d['text']))
        time_played_train.append(d['hours'])
    X_train = np.array(review_length_train).reshape(-1, 1)
    y_train = np.array(time_played_train)
    for d in dataTest:
        review_length_test.append(len(d['text']))
        time_played_test.append(d['hours'])
    X_test = np.array(review_length_test).reshape(-1, 1)
    y_test = np.array(time_played_test)
    model = LinearRegression()
    model.fit(X_train, y_train)
    theta0 = model.intercept_
    theta1 = model.coef_[0]
    y_pred = model.predict(X_test)
    mse = MSE(y_test, y_pred)
    cnt_under, cnt_over = reportOU(y_pred, y_test)
    return mse, cnt_under, cnt_over, theta0


def reportOU(y_pred, y_test):
    cnt_under = 0
    cnt_over = 0
    for p, g in zip(y_pred, y_test):
        if p > g:
            cnt_over += 1
        if p < g:
            cnt_under += 1
    return cnt_under, cnt_over


def Q3(dataset, theta0_q2):
    dataTrain = dataset[:int(len(dataset) * 0.8)]
    dataTest = dataset[int(len(dataset) * 0.8):]
    review_length_train = []
    time_played_train = []
    review_length_test = []
    time_played_test = []
    for d in dataTrain:
        review_length_train.append(len(d['text']))
        time_played_train.append(d['hours'])
    X_train = np.array(review_length_train).reshape(-1, 1)
    y_train = np.array(time_played_train)
    for d in dataTest:
        review_length_test.append(len(d['text']))
        time_played_test.append(d['hours'])
    X_test = np.array(review_length_test).reshape(-1, 1)
    y_test = np.array(time_played_test)
    # sub-question a
    data = list(zip(X_train, y_train))
    data_sorted = sorted(data, key=lambda x: x[1])
    data_sorted_perc90 = data_sorted[:int(len(data_sorted) * 0.9)]
    X_train_90, y_train_90 = zip(*data_sorted_perc90)
    model_a = LinearRegression()
    model_a.fit(X_train_90, y_train_90)
    y_pred_a = model_a.predict(X_test)
    cnt_under_a, cnt_over_a = reportOU(y_pred_a, y_test)
    # sub-question b
    time_played_train_trans = []
    time_played_test_trans = []
    for d in dataTrain:
        time_played_train_trans.append(d['hours_transformed'])
    for d in dataTest:
        time_played_test_trans.append(d['hours_transformed'])
    y_train_trans = np.array(time_played_train_trans)
    X_train_trans = X_train
    y_test_trans = np.array(time_played_test_trans)
    X_test_trans = X_test
    model_b = LinearRegression()
    model_b.fit(X_train_trans, y_train_trans)
    y_pred_b = model_b.predict(X_test_trans)
    cnt_under_b, cnt_over_b = reportOU(y_pred_b, y_test_trans)
    # sub-question c
    median_length = statistics.median(review_length_train)
    median_hours = statistics.median(time_played_train)
    theta_0 = theta0_q2
    theta_1 = (median_hours - theta_0) / median_length
    y_pred_c = [theta_0 + theta_1 * float(l) for l in review_length_test]
    cnt_under_c, cnt_over_c = reportOU(y_pred_c, y_test)
    return cnt_under_a, cnt_over_a, cnt_under_b, cnt_over_b, cnt_under_c, cnt_over_c


def Q4(dataset):
    dataTrain = dataset[:int(len(dataset) * 0.8)]
    dataTest = dataset[int(len(dataset) * 0.8):]
    review_length_train = []
    time_played_train = []
    review_length_test = []
    time_played_test = []
    for d in dataTrain:
        review_length_train.append(len(d['text']))
        time_played_train.append(d['hours'])
    X_train = np.array(review_length_train).reshape(-1, 1)
    medium_train = statistics.median(time_played_train)
    y_train = np.array([1 if i > medium_train else 0 for i in time_played_train])
    for d in dataTest:
        review_length_test.append(len(d['text']))
        time_played_test.append(d['hours'])
    X_test = np.array(review_length_test).reshape(-1, 1)
    medium_test = statistics.median(time_played_test)
    y_test = np.array([1 if i > medium_test else 0 for i in time_played_test])
    model = LogisticRegression(C=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    ber = BER(y_pred, y_test)
    return tp, tn, fp, fn, ber


def Q6(dataset):
    dataTrain = dataset[:int(len(dataset) * 0.8)]
    dataTest = dataset[int(len(dataset) * 0.8):]
    review_length_train_2014 = []
    time_played_train_2014 = []
    review_length_test_2014 = []
    time_played_test_2014 = []
    review_length_train_2015 = []
    time_played_train_2015 = []
    review_length_test_2015 = []
    time_played_test_2015 = []
    for d in dataTrain:
        year = int(d['date'][:4])
        if year <= 2014:
            review_length_train_2014.append(len(d['text']))
            time_played_train_2014.append(d['hours'])
        if year >= 2015:
            review_length_train_2015.append(len(d['text']))
            time_played_train_2015.append(d['hours'])
    X_train_2014 = np.array(review_length_train_2014).reshape(-1, 1)
    medium_train_2014 = statistics.median(time_played_train_2014)
    y_train_2014 = np.array([1 if i > medium_train_2014 else 0 for i in time_played_train_2014])
    X_train_2015 = np.array(review_length_train_2015).reshape(-1, 1)
    medium_train_2015 = statistics.median(time_played_train_2015)
    y_train_2015 = np.array([1 if i > medium_train_2015 else 0 for i in time_played_train_2015])
    for d in dataTest:
        year = int(d['date'][:4])
        if year <= 2014:
            review_length_test_2014.append(len(d['text']))
            time_played_test_2014.append(d['hours'])
        if year >= 2015:
            review_length_test_2015.append(len(d['text']))
            time_played_test_2015.append(d['hours'])
    X_test_2014 = np.array(review_length_test_2014).reshape(-1, 1)
    medium_test_2014 = statistics.median(time_played_test_2014)
    y_test_2014 = np.array([1 if i > medium_test_2014 else 0 for i in time_played_test_2014])
    X_test_2015 = np.array(review_length_test_2015).reshape(-1, 1)
    medium_test_2015 = statistics.median(time_played_test_2015)
    y_test_2015 = np.array([1 if i > medium_test_2015 else 0 for i in time_played_test_2015])
    model_a = LogisticRegression(C=1)
    model_a.fit(X_train_2014, y_train_2014)
    y_pred_a = model_a.predict(X_test_2014)
    tn, fp, fn, tp = confusion_matrix(y_test_2014, y_pred_a).ravel()
    ber_a = 0.5 * (fn / (fn + tp) + fp / (fp + tn))

    model_b = LogisticRegression(C=1)
    model_b.fit(X_train_2015, y_train_2015)
    y_pred_b = model_b.predict(X_test_2015)
    tn, fp, fn, tp = confusion_matrix(y_test_2015, y_pred_b).ravel()
    ber_b = 0.5 * (fn / (fn + tp) + fp / (fp + tn))

    model_c = LogisticRegression(C=1)
    model_c.fit(X_train_2014, y_train_2014)
    y_pred_c = model_c.predict(X_test_2015)
    tn, fp, fn, tp = confusion_matrix(y_test_2015, y_pred_c).ravel()
    ber_c = 0.5 * (fn / (fn + tp) + fp / (fp + tn))

    model_a = LogisticRegression(C=1)
    model_a.fit(X_train_2015, y_train_2015)
    y_pred_d = model_a.predict(X_test_2014)
    tn, fp, fn, tp = confusion_matrix(y_test_2014, y_pred_d).ravel()
    ber_d = 0.5 * (fn / (fn + tp) + fp / (fp + tn))
    return ber_a, ber_b, ber_c, ber_d


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


def Q7(dataset):
    dataTrain = dataset[:int(len(dataset) * 0.8)]
    dataTest = dataset[int(len(dataset) * 0.8):]
    usersPerItem = defaultdict(set)  # Maps an item to the users who rated it
    itemsPerUser = defaultdict(set)
    timeDict = {}
    for d in dataTrain:
        user,item = d['userID'], d['gameID']
        usersPerItem[item].add(user)
        itemsPerUser[user].add(item)
        timeDict[(user,item)] = d['hours_transformed']
    similarities = []
    items_0 = itemsPerUser[dataTrain[0]['userID']]
    for i in itemsPerUser:
        if i == dataTrain[0]['userID']:
            continue
        items_cur = itemsPerUser[i]
        sim = Jaccard(items_0,items_cur)
        similarities.append((sim,i))
    similarities = sorted(similarities, key=lambda x: x[0], reverse=True)[:10]
    return similarities[0][0],similarities[9][0]
def Q8(dataset):
    dataTrain = dataset[:int(len(dataset) * 0.8)]
    dataTest = dataset[int(len(dataset) * 0.8):]
    usersPerItem = defaultdict(set)  # Maps an item to the users who rated it
    itemsPerUser = defaultdict(set)
    timeDict = {}
    for d in dataTrain:
        user, item = d['userID'], d['gameID']
        usersPerItem[item].add(user)
        itemsPerUser[user].add(item)
        timeDict[(user, item)] = d['hours_transformed']
    timeMean = sum([d['hours_transformed'] for d in dataset]) / len(dataset)

    def predictTimeUU(user, item):
        times = []
        similarities = []
        if item not in usersPerItem.keys():
            return timeMean
        for d in usersPerItem[item]:
            cur_user = d
            if cur_user == user: continue
            times.append(timeDict[(cur_user,item)])
            similarities.append(Jaccard(itemsPerUser[cur_user],itemsPerUser[user]))
        if (sum(similarities) > 0):
            weightedRatings = [(x * y) for x, y in zip(times, similarities)]
            return sum(weightedRatings) / sum(similarities)
        else:
            return timeMean

    predictions, labels = [], []
    for d in tqdm.tqdm(dataTest):
        user_id, game_id, time = d['userID'], d['gameID'], d['hours_transformed']
        predictions.append(predictTimeUU(user_id, game_id))
        labels.append(time)
    mse_uu = MSE(predictions, labels)


    def predictTimeII(user, item):
        times = []
        similarities = []
        if user not in itemsPerUser.keys():
            return timeMean
        for d in itemsPerUser[user]:
            cur_item = d
            if cur_item == item: continue
            times.append(timeDict[(user,cur_item)])
            similarities.append(Jaccard(usersPerItem[item], usersPerItem[cur_item]))
        if (sum(similarities) > 0):
            weightedRatings = [(x * y) for x, y in zip(times, similarities)]
            return sum(weightedRatings) / sum(similarities)
        else:
            return timeMean

    predictions, labels = [], []
    for d in tqdm.tqdm(dataTest):
        user_id, game_id, time = d['userID'], d['gameID'], d['hours_transformed']
        predictions.append(predictTimeII(user_id, game_id))
        labels.append(time)
    mse_ii = MSE(predictions, labels)
    return mse_uu,mse_ii
def Q9(dataset):
    dataTrain = dataset[:int(len(dataset) * 0.8)]
    dataTest = dataset[int(len(dataset) * 0.8):]
    usersPerItem = defaultdict(set)  # Maps an item to the users who rated it
    itemsPerUser = defaultdict(set)
    timeDict = {}
    yearDict = {}
    for d in dataTrain:
        user, item = d['userID'], d['gameID']
        usersPerItem[item].add(user)
        itemsPerUser[user].add(item)
        timeDict[(user, item)] = d['hours_transformed']
        yearDict[(user,item)] = int(d['date'][:4])
    for d in dataTest:
        user, item = d['userID'], d['gameID']
        yearDict[(user, item)] = int(d['date'][:4])
    timeMean = sum([d['hours_transformed'] for d in dataset]) / len(dataset)

    def predictTimeUU(user, item):
        times = []
        similarities = []
        if item not in usersPerItem.keys():
            return timeMean
        for d in usersPerItem[item]:
            cur_user = d
            if cur_user == item: continue
            times.append(timeDict[(cur_user,item)])
            sim = Jaccard(itemsPerUser[cur_user],itemsPerUser[user])
            exp = math.exp(-abs(yearDict[(user,item)]-yearDict[(cur_user,item)]))
            similarities.append(sim*exp)
        if (sum(similarities) > 0):
            weightedRatings = [(x * y) for x, y in zip(times, similarities)]
            return sum(weightedRatings) / sum(similarities)
        else:
            return timeMean

    predictions, labels = [], []
    for d in tqdm.tqdm(dataTest):
        user_id, game_id, time = d['userID'], d['gameID'], d['hours_transformed']
        predictions.append(predictTimeUU(user_id, game_id))
        labels.append(time)
    mse = MSE(predictions, labels)
    return mse
if __name__ == '__main__':
    dataset = readDataset()
    theta1, mse1 = Q1(dataset)
    answers['Q1'] = [float(theta1), float(mse1)]
    assertFloatList(answers['Q1'], 2)
    mse2, under, over, theta0_q2 = Q2(dataset)
    answers['Q2'] = [float(mse2), float(under), float(over)]
    assertFloatList(answers['Q2'], 3)
    under3a, over3a, under3b, over3b, under3c, over3c = Q3(dataset, theta0_q2)
    answers['Q3'] = [float(under3a), float(over3a), float(under3b), float(over3b), float(under3c), float(over3c)]
    assertFloatList(answers['Q3'], 6)
    TP, TN, FP, FN, BER = Q4(dataset)
    answers['Q4'] = [TP, TN, FP, FN, BER]
    assertFloatList(answers['Q4'], 5)
    answers['Q5'] = [FP, FN]
    assertFloatList(answers['Q5'], 2)
    BER_A, BER_B, BER_C, BER_D = Q6(dataset)
    answers['Q6'] = [BER_A, BER_B, BER_C, BER_D]
    assertFloatList(answers['Q6'], 4)
    first, tenth = Q7(dataset)
    answers['Q7'] = [first, tenth]
    assertFloatList(answers['Q7'], 2)
    MSEU, MSEI = Q8(dataset)
    answers['Q8'] = [MSEU, MSEI]
    assertFloatList(answers['Q8'], 2)
    MSE9 = Q9(dataset)
    answers['Q9'] = MSE9
    assertFloat(answers['Q9'])
    f = open("answers_midterm.txt", 'w')
    f.write(str(answers) + '\n')
    f.close()
