import gzip
import time
from collections import defaultdict
import math

import numpy as np
import scipy.optimize
import tqdm
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import warnings
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm.contrib import tzip

warnings.filterwarnings("ignore")
random.seed(42)


def assertFloat(x):
    assert type(float(x)) == float


def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float] * N


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u, b, r = l.strip().split(',')
        r = int(r)
        yield u, b, r


def randomSelect(user, all_valid_book, ratingsPerUser):
    books_read = set([i[0] for i in ratingsPerUser[user]])
    difference = all_valid_book - books_read
    random_element = random.choice(list(difference))
    return random_element


def Q1():
    allRatings = []
    for l in readCSV("train_Interactions.csv.gz"):
        allRatings.append(l)
    ratingsTrain = allRatings[:190000]
    ratingsValid = allRatings[190000:]
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    for u, b, r in ratingsTrain:
        ratingsPerUser[u].append((b, r))
        ratingsPerItem[b].append((u, r))
    all_valid_user = set()
    all_valid_book = set()
    for r in ratingsValid:
        all_valid_user.add(r[0])
        all_valid_book.add(r[1])
    valid_set = []
    for r in ratingsValid:
        valid_set.append((r[0], r[1], 1))
        valid_set.append((r[0], randomSelect(r[0], all_valid_book, ratingsPerUser), 0))
    bookCount = defaultdict(int)
    totalRead = 0

    for user, book, _ in readCSV("train_Interactions.csv.gz"):
        bookCount[book] += 1
        totalRead += 1

    mostPopular = [(bookCount[x], x) for x in bookCount]
    mostPopular.sort()
    mostPopular.reverse()

    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead / 2: break
    pred = []
    gold = [i[2] for i in valid_set]
    for v in valid_set:
        cur_book = v[1]
        if cur_book in return1:
            pred.append(1)
        else:
            pred.append(0)
    res = [p == g for p, g in zip(pred, gold)]
    print(sum(res) / len(res))
    return sum(res) / len(res)


def Q2():
    allRatings = []
    for l in readCSV("train_Interactions.csv.gz"):
        allRatings.append(l)
    ratingsTrain = allRatings[:190000]
    ratingsValid = allRatings[190000:]
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    for u, b, r in ratingsTrain:
        ratingsPerUser[u].append((b, r))
        ratingsPerItem[b].append((u, r))
    all_valid_user = set()
    all_valid_book = set()
    for r in ratingsValid:
        all_valid_user.add(r[0])
        all_valid_book.add(r[1])
    valid_set = []
    for r in ratingsValid:
        valid_set.append((r[0], r[1], 1))
        valid_set.append((r[0], randomSelect(r[0], all_valid_book, ratingsPerUser), 0))
    bookCount = defaultdict(int)
    totalRead = 0

    for user, book, _ in readCSV("train_Interactions.csv.gz"):
        bookCount[book] += 1
        totalRead += 1

    mostPopular = [(bookCount[x], x) for x in bookCount]
    mostPopular.sort()
    mostPopular.reverse()

    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead / 2: break
    pred = []
    gold = [i[2] for i in valid_set]
    for v in valid_set:
        cur_book = v[1]
        if cur_book in return1:
            pred.append(1)
        else:
            pred.append(0)
    res = [p == g for p, g in zip(pred, gold)]
    accu = sum(res) / len(res)
    best_accu = accu
    best_threshold = 0
    accus = []
    for threshold in tqdm.tqdm(range(45, 75)):
        cur_return = set()
        count = 0
        for ic, i in mostPopular:
            count += ic
            cur_return.add(i)
            if count > totalRead * threshold / 100: break
        pred = []
        gold = [i[2] for i in valid_set]
        for v in valid_set:
            cur_book = v[1]
            if cur_book in cur_return:
                pred.append(1)
            else:
                pred.append(0)
        res = [p == g for p, g in zip(pred, gold)]
        cur_accu = sum(res) / len(res)
        accus.append(cur_accu)
        if cur_accu > best_accu:
            best_accu = cur_accu
            best_threshold = threshold

    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(45, 75)], accus, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold')
    plt.grid()
    # Save the training accuracy figure
    training_accuracy_file = 'train_accuracy_Q2.png'
    plt.savefig(training_accuracy_file)
    print("{} {}".format(best_accu, best_threshold))
    return best_threshold, best_accu


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


def Q3():
    allRatings = []
    for l in readCSV("train_Interactions.csv.gz"):
        allRatings.append(l)
    ratingsTrain = allRatings[:190000]
    ratingsValid = allRatings[190000:]
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    booksPerUser = defaultdict(set)
    usersPerBook = defaultdict(set)
    for u, b, r in ratingsTrain:
        ratingsPerUser[u].append((b, r))
        ratingsPerItem[b].append((u, r))
        booksPerUser[u].add(b)
        usersPerBook[b].add(u)
    all_valid_user = set()
    all_valid_book = set()
    for r in ratingsValid:
        all_valid_user.add(r[0])
        all_valid_book.add(r[1])
    valid_set = []
    for r in ratingsValid:
        valid_set.append((r[0], r[1], 1))
        valid_set.append((r[0], randomSelect(r[0], all_valid_book, ratingsPerUser), 0))

    gold = [i[2] for i in valid_set]

    # Grid search for the best threshold
    best_accuracy = 0
    best_threshold = 0
    accus = []
    for threshold in tqdm.tqdm([i * 0.01 for i in range(10)]):  # Test thresholds from 0 to 1 with step 0.01
        pred = []
        for v in valid_set:
            u = v[0]
            b = v[1]
            bprimes = booksPerUser[u]
            b_set = usersPerBook[b]
            jaccard = 0
            for bprime in bprimes:
                bprime_set = usersPerBook[bprime]
                jaccard = max(jaccard, Jaccard(b_set, bprime_set))
            if jaccard > threshold:
                pred.append(1)
            else:
                pred.append(0)
        res = [p == g for p, g in zip(pred, gold)]
        accuracy = sum(res) / len(res)
        accus.append(accuracy)
        # Update the best threshold if the current one is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(10)], accus, marker='o', linestyle='-', linewidth=2, markersize=6)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold')
    plt.grid()
    # Save the training accuracy figure
    training_accuracy_file = 'train_accuracy_Q3.png'
    plt.savefig(training_accuracy_file)
    print(f"Best threshold: {best_threshold}, Accuracy: {best_accuracy}")
    return best_accuracy


def Q4():
    allRatings = []
    for l in readCSV("train_Interactions.csv.gz"):
        allRatings.append(l)
    ratingsTrain = allRatings[:190000]
    ratingsValid = allRatings[190000:]
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    booksPerUser = defaultdict(set)
    usersPerBook = defaultdict(set)
    bookCount = defaultdict(int)
    totalRead = 0

    for u, b, r in ratingsTrain:
        ratingsPerUser[u].append((b, r))
        ratingsPerItem[b].append((u, r))
        booksPerUser[u].add(b)
        usersPerBook[b].add(u)

    all_valid_user = set()
    all_valid_book = set()
    for r in ratingsValid:
        all_valid_user.add(r[0])
        all_valid_book.add(r[1])
    valid_set = []
    for r in ratingsValid:
        valid_set.append((r[0], r[1], 1))
        valid_set.append((r[0], randomSelect(r[0], all_valid_book, ratingsPerUser), 0))
    for user, book, _ in readCSV("train_Interactions.csv.gz"):
        bookCount[book] += 1
        totalRead += 1
    mostPopular = [(bookCount[x], x) for x in bookCount]
    mostPopular.sort(reverse=True)
    returns = {}
    for popularity_threshold in range(45, 75):
        cur_return = set()
        count = 0
        for ic, i in mostPopular:
            count += ic
            cur_return.add(i)
            if count > totalRead * popularity_threshold / 100:
                break
        returns[popularity_threshold] = cur_return
    jaccards = []
    for v in tqdm.tqdm(valid_set):
        u = v[0]
        b = v[1]

        # Jaccard calculation
        bprimes = booksPerUser[u]
        b_set = usersPerBook[b]
        jaccard = 0
        for bprime in bprimes:
            bprime_set = usersPerBook[bprime]
            jaccard = max(jaccard, Jaccard(b_set, bprime_set))
        jaccards.append([b, jaccard])

    gold = [i[2] for i in valid_set]

    # Grid search for the best Jaccard and popularity thresholds
    best_accuracy = 0
    best_jaccard_threshold = 0
    best_popularity_threshold = 0
    for popularity_threshold in range(45, 75):
        cur_return = returns[popularity_threshold]
        for jaccard_threshold in [i * 0.01 for i in range(10)]:
            pred = []
            for t in jaccards:
                b = t[0]
                jaccard = t[1]
                if jaccard > jaccard_threshold or b in cur_return:
                    pred.append(1)
                else:
                    pred.append(0)
            res = [p == g for p, g in zip(pred, gold)]
            accuracy = sum(res) / len(res)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_jaccard_threshold = jaccard_threshold
                best_popularity_threshold = popularity_threshold
    print(
        f"Best Jaccard threshold: {best_jaccard_threshold}, Best popularity threshold: {best_popularity_threshold}, Accuracy: {best_accuracy}")
    return best_accuracy


def Q5():
    jaccard_threshold = 0.02
    popularity_threshold = 59
    allRatings = []
    for l in readCSV("train_Interactions.csv.gz"):
        allRatings.append(l)
    ratingsTrain = allRatings[:190000]
    booksPerUser = defaultdict(set)
    usersPerBook = defaultdict(set)
    bookCount = defaultdict(int)
    totalRead = 0
    for u, b, r in ratingsTrain:
        booksPerUser[u].add(b)
        usersPerBook[b].add(u)
    for user, book, _ in readCSV("train_Interactions.csv.gz"):
        bookCount[book] += 1
        totalRead += 1
    mostPopular = [(bookCount[x], x) for x in bookCount]
    mostPopular.sort(reverse=True)
    cur_return = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        cur_return.add(i)
        if count > totalRead * popularity_threshold / 100:
            break
    predictions = open("predictions_Read.csv", 'w')
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, b = l.strip().split(',')
        bprimes = booksPerUser[u]
        b_set = usersPerBook[b]
        jaccard = 0
        for bprime in bprimes:
            bprime_set = usersPerBook[bprime]
            jaccard = max(jaccard, Jaccard(b_set, bprime_set))
        if jaccard > jaccard_threshold or b in cur_return:
            predictions.write(u + ',' + b + ",1\n")
        else:
            predictions.write(u + ',' + b + ",0\n")
    predictions.close()
    return


def map_ids(ratings):
    user_map = {}
    item_map = {}
    user_count = 0
    item_count = 0

    for u, b, _ in ratings:
        if u not in user_map:
            user_map[u] = user_count
            user_count += 1
        if b not in item_map:
            item_map[b] = item_count
            item_count += 1

    return user_map, item_map, user_count, item_count


def train_bias_model(ratings, user_map, item_map, user_count, item_count, lambda_reg=1, epochs=100, lr=0.01):
    # 初始化参数
    alpha = np.mean([float(r) for _, _, r in ratings])  # 全局平均评分
    beta_user = np.zeros(user_count)  # 用户偏置
    beta_item = np.zeros(item_count)  # 物品偏置

    for epoch in tqdm.tqdm(range(epochs)):
        # 计算损失
        sse = 0
        for u, b, rating in ratings:
            user_idx = user_map[u]
            item_idx = item_map[b]
            rating = float(rating)

            prediction = alpha + beta_user[user_idx] + beta_item[item_idx]
            error = rating - prediction
            sse += error ** 2

            # 梯度更新
            beta_user[user_idx] += lr * (error - lambda_reg * beta_user[user_idx])
            beta_item[item_idx] += lr * (error - lambda_reg * beta_item[item_idx])

        # 正则化部分
        reg_term = lambda_reg * (np.sum(beta_user ** 2) + np.sum(beta_item ** 2))
        total_loss = sse + reg_term

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss}")

    return alpha, beta_user, beta_item


def predict(user, item, alpha, beta_user, beta_item, user_map, item_map):
    # 用户和物品都存在于映射中
    if user in user_map and item in item_map:
        user_idx = user_map[user]
        item_idx = item_map[item]
        return alpha + beta_user[user_idx] + beta_item[item_idx]
    # 如果用户未知但物品已知
    elif item in item_map:
        item_idx = item_map[item]
        return alpha + beta_item[item_idx]
    # 如果物品未知但用户已知
    elif user in user_map:
        user_idx = user_map[user]
        return alpha + beta_user[user_idx]
    # 如果用户和物品都未知
    else:
        return alpha


def calculate_mse(ratings, alpha, beta_user, beta_item, user_map, item_map):
    errors = [(float(rating) - predict(user, item, alpha, beta_user, beta_item, user_map, item_map)) ** 2
              for user, item, rating in ratings]
    return np.mean(errors)


def Q6():
    allRatings = []
    for l in readCSV("train_Interactions.csv.gz"):
        allRatings.append(l)
    ratingsTrain = allRatings[:190000]
    ratingsValid = allRatings[190000:]
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    for u, b, r in ratingsTrain:
        ratingsPerUser[u].append((b, r))
        ratingsPerItem[b].append((u, r))

    user_map, item_map, user_count, item_count = map_ids(ratingsTrain)
    alpha, beta_user, beta_item = train_bias_model(ratingsTrain, user_map, item_map, user_count, item_count)
    mse = calculate_mse(ratingsValid, alpha, beta_user, beta_item, user_map, item_map)
    return mse


def Q7():
    allRatings = []
    for l in readCSV("train_Interactions.csv.gz"):
        allRatings.append(l)
    ratingsTrain = allRatings[:190000]
    ratingsValid = allRatings[190000:]
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    for u, b, r in ratingsTrain:
        ratingsPerUser[u].append((b, r))
        ratingsPerItem[b].append((u, r))

    user_map, item_map, user_count, item_count = map_ids(ratingsTrain)
    alpha, beta_user, beta_item = train_bias_model(ratingsTrain, user_map, item_map, user_count, item_count)
    max_beta_value = np.max(beta_user)
    min_beta_value = np.min(beta_user)
    max_user_idx = np.argmax(beta_user)
    min_user_idx = np.argmin(beta_user)

    # 通过映射找到对应的用户ID
    max_user_id = [user for user, idx in user_map.items() if idx == max_user_idx][0]
    min_user_id = [user for user, idx in user_map.items() if idx == min_user_idx][0]

    return max_user_id, min_user_id, float(max_beta_value), float(min_beta_value)


def Q8():
    lambda_values = [0.01, 0.1, 0.5, 1, 2, 5, 10]
    best_lambda = None
    best_mse = float('inf')
    allRatings = []
    for l in readCSV("train_Interactions.csv.gz"):
        allRatings.append(l)
    ratingsTrain = allRatings[:190000]
    ratingsValid = allRatings[190000:]
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    for u, b, r in ratingsTrain:
        ratingsPerUser[u].append((b, r))
        ratingsPerItem[b].append((u, r))

    user_map, item_map, user_count, item_count = map_ids(ratingsTrain)
    best_alpha, best_beta_user, best_beta_item = None,None,None
    for lambda_reg in lambda_values:
        print(f"Training with λ = {lambda_reg}")
        alpha, beta_user, beta_item = train_bias_model(ratingsTrain, user_map, item_map, user_count, item_count,
                                                       lambda_reg=lambda_reg, epochs=100, lr=0.01)
        mse = calculate_mse(ratingsValid, alpha, beta_user, beta_item, user_map, item_map)
        print(f"Validation MSE for λ = {lambda_reg}: {mse}")

        if mse < best_mse:
            best_mse = mse
            best_lambda = lambda_reg
            best_alpha, best_beta_user, best_beta_item = alpha, beta_user, beta_item

    predictions = open("predictions_Rating.csv", 'w')
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            # header
            predictions.write(l)
            continue
        u, b = l.strip().split(',')
        rating = predict(u, b, best_alpha, best_beta_user, best_beta_item, user_map, item_map)
        predictions.write(u + ',' + b + ',' + str(rating) + '\n')
    predictions.close()

    return best_lambda,best_mse


def main():
    answers = {}
    answers['Q1'] = Q1()
    assertFloat(answers['Q1'])
    threshold, acc2 = Q2()
    answers['Q2'] = [threshold, acc2]
    assertFloat(answers['Q2'][0])
    assertFloat(answers['Q2'][1])
    answers['Q3'] = Q3()
    answers['Q4'] = Q4()
    assertFloat(answers['Q3'])
    assertFloat(answers['Q4'])
    Q5()
    answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"
    validMSE = Q6()
    answers['Q6'] = validMSE
    assertFloat(answers['Q6'])
    maxUser, minUser, maxBeta, minBeta = Q7()
    answers['Q7'] = [maxUser, minUser, maxBeta, minBeta]
    assert [type(x) for x in answers['Q7']] == [str, str, float, float]
    lamb, validMSE = Q8()
    answers['Q8'] = (lamb, validMSE)
    assertFloat(answers['Q8'][0])
    assertFloat(answers['Q8'][1])
    f = open("answers_hw3.txt", 'w')
    f.write(str(answers) + '\n')
    f.close()


if __name__ == '__main__':
    main()
