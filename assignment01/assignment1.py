import gzip
import random
from collections import defaultdict

import numpy as np
import tqdm


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u, b, r = l.strip().split(',')
        r = int(r)
        yield u, b, r


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom
def randomSelect(user, all_valid_book, ratingsPerUser):
    books_read = set([i[0] for i in ratingsPerUser[user]])
    difference = all_valid_book - books_read
    random_element = random.choice(list(difference))
    return random_element

def trainPredict():
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

    all_valid_user = set(r[0] for r in ratingsValid)
    all_valid_book = set(r[1] for r in ratingsValid)
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
        returns[popularity_threshold] = frozenset(cur_return)  # frozenset for faster lookup

    def adaptive_jaccard_threshold(user, book, base_threshold):
        user_adjustment = 1 + len(booksPerUser[user]) * 0.01  # 用户活跃度调整
        book_adjustment = 1 - min(bookCount[book] / totalRead, 0.5)  # 书籍流行度调整
        return base_threshold * user_adjustment * book_adjustment

    jaccard_cache = {}
    jaccards = []
    base_jaccard_threshold = 0.02
    for v in tqdm.tqdm(valid_set):
        u, b = v[0], v[1]
        if (u, b) not in jaccard_cache:
            bprimes = booksPerUser[u]
            b_set = usersPerBook[b]
            jaccard = 0
            threshold = adaptive_jaccard_threshold(u, b, base_jaccard_threshold)
            for bprime in bprimes:
                bprime_set = usersPerBook[bprime]
                jaccard = max(jaccard, Jaccard(b_set, bprime_set))
                if jaccard >= threshold:  # 提前停止
                    break
            jaccard_cache[(u, b)] = jaccard
        else:
            jaccard = jaccard_cache[(u, b)]
        jaccards.append([b, jaccard])

    gold = [i[2] for i in valid_set]

    # Parallelize grid search for efficiency
    best_accuracy = 0
    best_jaccard_threshold = 0
    best_popularity_threshold = 0
    for popularity_threshold in range(45, 75):
        cur_return = returns[popularity_threshold]
        for jaccard_threshold in [i * 0.02 for i in range(5)]:  # Adjusted step size
            pred = []
            for t in jaccards:
                b, jaccard = t
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
    print(f"Best Jaccard threshold: {best_jaccard_threshold}, Best popularity threshold: {best_popularity_threshold}, Accuracy: {best_accuracy}")
    return best_accuracy




def readPredict():
    jaccard_threshold = 0.02
    popularity_threshold = 61
    adaptive_jaccard = True  # 自适应Jaccard阈值标志
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
    def adaptive_jaccard_threshold(user, book, base_threshold):
        user_adjustment = 1 + len(booksPerUser[user]) * 0.01
        book_adjustment = 1 - min(bookCount[book] / totalRead, 0.5)
        return base_threshold * user_adjustment * book_adjustment
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, b = l.strip().split(',')
        bprimes = booksPerUser[u]
        b_set = usersPerBook[b]
        jaccard = 0
        threshold = jaccard_threshold
        if adaptive_jaccard:
            threshold = adaptive_jaccard_threshold(u, b, jaccard_threshold)
        for bprime in bprimes:
            bprime_set = usersPerBook[bprime]
            jaccard = max(jaccard, Jaccard(b_set, bprime_set))
        if jaccard > threshold or b in cur_return:
            predictions.write(u + ',' + b + ",1\n")
        else:
            predictions.write(u + ',' + b + ",0\n")
    predictions.close()
    return


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


def calculate_mse(ratings, alpha, beta_user, beta_item, user_map, item_map):
    errors = [(float(rating) - predict(user, item, alpha, beta_user, beta_item, user_map, item_map)) ** 2
              for user, item, rating in ratings]
    return np.mean(errors)


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

def readRating():
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
    best_alpha, best_beta_user, best_beta_item = None, None, None
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

if __name__ == '__main__':
    readPredict()
    readRating()