import os
import random
import time
import re
import numpy as np
import codecs
import math
import time
import string
import sys
import xxhash
from collections import defaultdict

stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])


g_classes = np.array(['American_comedy_films', 'American_drama_films', 'American_film_actresses', 'American_film_directors', 'American_films', 'American_male_film_actors', 'American_male_television_actors', 'American_military_personnel_of_World_War_II', 'American_people_of_Irish_descent', 'American_television_actresses', 'Arctiidae', 'Articles_containing_video_clips', 'Association_football_defenders', 'Association_football_forwards', 'Association_football_goalkeepers', 'Association_football_midfielders', 'Asteroids_named_for_people', 'Australian_rules_footballers_from_Victoria_(Australia)', 'Black-and-white_films', 'Brazilian_footballers', 'British_films', 'Columbia_University_alumni', 'Deaths_from_myocardial_infarction', 'English_cricketers', 'English_footballers', 'English-language_albums', 'English-language_films', 'English-language_journals', 'English-language_television_programming', 'Fellows_of_the_Royal_Society', 'French_films', 'German_footballers', 'Guggenheim_Fellows', 'Harvard_University_alumni', 'Hindi-language_films', 'Indian_films', 'Insects_of_Europe', 'Italian_films', 'Italian_footballers', 'Main_Belt_asteroids', 'Major_League_Baseball_pitchers', 'Rivers_of_Romania', 'Russian_footballers', 'Scottish_footballers', 'Serie_A_players', 'The_Football_League_players', 'Villages_in_the_Czech_Republic', 'Villages_in_Turkey', 'Windows_games', 'Yale_University_alumni'])

def getLabelsAndWords(line):
    labels = []
    words = []
    tokens = line.split(sep = '\t')
    # print('tokens = ', tokens)
    tmp = tokens[0].split(',')
    # print('labels = ', tmp)
    for c in tmp:
        labels.append(c.strip())
    idx_begin = tokens[1].index("\"")
    idx_end = tokens[1].rindex("\"")
    document = tokens[1][idx_begin+1:idx_end]
    wrds = re.findall('\\w+', document)
    # translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    # document = document.translate(translator)
    # document = document.split()
    for d in wrds:
        # d = re.sub(r'[^\w\s]','', d)
        d = d.lower()
        # if d not in stopwords:
        words.append(d)
    return set(labels), words

def sigmoid(x):
	overflow = 20.0
	if x > overflow:
		x = overflow
	elif x < -overflow:
		x = -overflow
	ex = math.exp(x)
	return ex / (1 + ex)

def getNZFeatureIndices(words, N):
    b_idx = np.array([xxhash.xxh64(w).intdigest()%N for w in words])
    return b_idx


def incLR(init_lr, n_epoch):
    return init_lr

def decLR(init_lr, n_epoch):
    return init_lr/(n_epoch*n_epoch)

def constantLR(init_lr, n_epoch):
    return init_lr

def trainLR(classes, classesToIdxMap, n_vocab, init_lr, mu, n_epochs, n_examples, update_lr):
    n_classes = len(classes)
    # print('n_classes = ', n_classes)
    # The output params for each of the n_classes
    # Used for lazy evaluation of \beta
    AA = np.zeros((n_vocab, n_classes))
    BB = np.zeros((n_vocab, n_classes))
    LCL = np.zeros(n_epochs)
    k = 0
    old_t = 1
    for l in sys.stdin:
        t = 1 + k//n_examples
        if (t != old_t):
            # A new epoch has started
            old_t = t
            for c in range(n_classes):
                BB[:, c] = BB[:, c]*((1 - 2*mu*lr)**(k - AA[:, c]))
                AA[:, c] = k
            print('Epoch :', t, "New learning rate =", update_lr(init_lr, t))

        # t now contains the current epoch value
        k += 1
        # New learning rate
        lr = update_lr(init_lr, t)
        # print('lr = ', lr)
        if (k%1000 == 0):
            print('Epoch:', t, 'k =', k)

        labels, words = getLabelsAndWords(l)
        if (len(labels) <= 0 or len(words) <= 0):
            continue
        # print('labels = ', labels)
        # print('words = ', words)
        index_b = getNZFeatureIndices(words, n_vocab)
        if len(index_b) <= 0:
            print('index_b = ', index_b, 'words =', words, 'labels =', labels, 'k = ', k)

        yy = np.zeros(n_classes)
        for label in labels:
            yy[classesToIdxMap[label]] = 1
            # print('label = ', label, 'k = ', k)
        for c in range(n_classes):
            # print('updating Beta for class', classes[c])
            # B = BB[:, c]
            # print('initial BB = ', BB[:,c])
            # A = AA[:, c]
            # print('initial AA = ', AA[:,c])
            BB[index_b, c] = BB[index_b, c]*((1 - 2*mu*lr)**(k - AA[index_b, c] - 1))
            p = sigmoid(np.sum(BB[index_b, c]))
            # print('p = ', p)
            BB[index_b, c] = BB[index_b, c]*(1 - 2*mu*lr)
            y = yy[c]
            BB[index_b, c] += lr*(y - p)
            AA[index_b, c] = k
            if y == 0:
                LCL[t-1] += math.log(1-p)
            elif y == 1:
                LCL[t-1] += math.log(p)
            else:
                assert False

            # print('final B = ', BB[:, c])
            # print('final A = ', AA[:, c])

    # One last update(lazy)
    # print('before final update BB = ', BB)
    # print('before final update AA = ', AA)
    for c in range(n_classes):
        BB[:, c] = BB[:, c]*((1 - 2*mu*lr)**(k - AA[:, c]))
        AA[:, c] = k

    # print('After final update AA = ', AA)
    # print('After final update BB = ', BB)

    print('Final BB = ', BB)
    # Now we have updated the Betas of all the classes using this example
    return BB, LCL

def predict(BB, words):
    n_vocab = BB.shape[0]
    n_classes = BB.shape[1]
    index_b = getNZFeatureIndices(words, n_vocab)
    proba = np.empty(n_classes)
    for c in range(n_classes):
        B = BB[:, c]
        proba[c] = sigmoid(np.sum(B[index_b]))
    return proba


def testLR(path, BB, classes):
    f = open(path, 'r')
    lines = f.readlines()
    errors = 0
    correct = 0
    total = 0

    for l in lines:
        trueClass, words = getLabelsAndWords(l)
        if (len(trueClass) <= 0 or len(words) <= 0):
            errors += 1
            total += 1
            continue
        proba = predict(BB, words)
        idx = (proba < 0.5)
        incorrectPrediction = True
        if np.all(idx):
            # print(trueClass, '|')
            pass
        else:
            predictedClass = classes[~idx]
            # print(trueClass, '|', predictedClass, '||', proba[~idx])
            for c in predictedClass:
                if c in trueClass:
                    correct += 1
                    incorrectPrediction = False
                    break
        if incorrectPrediction == True:
            errors +=1
        total += 1
        if total % 10000 == 0:
            print('Correct = ', correct, 'Total predictions = ', total)
            print('Accuracy = ', correct/total)
    assert (total == errors + correct)
    accuracy = correct/total
    return accuracy

arg_len = len(sys.argv)
if (arg_len != 7):
    print('Usage: python3 lr_inmemory.py <vocabularySize> <initial_learning_rate> <regularizationCoeff> <n_epochs> <n_examples> <test_file_path>')
    sys.exit(3)
vocabularySize = int(sys.argv[1])
learningRate = float(sys.argv[2])
regularizationCoeff = float(sys.argv[3])
n_epochs = int(sys.argv[4])
n_examples = int(sys.argv[5])

print('(vocabularySize, initial_learning_rate, regularizationCoeff, n_epochs, n_examples) = (', vocabularySize,
        learningRate, regularizationCoeff, n_epochs, n_examples, sys.argv[6], ')')

print('Training...')
t1 = time.time()
classesToIdxMap = {}
for i in range(len(g_classes)):
    classesToIdxMap[g_classes[i]] = i

assert len(classesToIdxMap) == len(g_classes)
# print(classesToIdxMap)
params, LCL = trainLR(g_classes, classesToIdxMap, vocabularySize, learningRate, regularizationCoeff, n_epochs, n_examples, decLR)
t2 = time.time()
time_to_train = t2 - t1
print('Training time = ', time_to_train, 'seconds')
for i in range(n_epochs):
    print('LCL for epoch', i+1, '=', LCL[i])
print('Testing...')
t1 = time.time()
accuracy = testLR(sys.argv[6], params, g_classes)
t2 = time.time()
time_to_test = t2 - t1
print('Final accuracy = ', accuracy)
print('Testing time = ', time_to_test, 'seconds')
