import numpy as np
import tensorflow as tf
import re
import os
import random
import time
import codecs
import math
import time
import string
import sys
import xxhash

g_classes = np.array(['American_comedy_films', 'American_drama_films', 'American_film_actresses', 'American_film_directors', 'American_films', 'American_male_film_actors', 'American_male_television_actors', 'American_military_personnel_of_World_War_II', 'American_people_of_Irish_descent', 'American_television_actresses', 'Arctiidae', 'Articles_containing_video_clips', 'Association_football_defenders', 'Association_football_forwards', 'Association_football_goalkeepers', 'Association_football_midfielders', 'Asteroids_named_for_people', 'Australian_rules_footballers_from_Victoria_(Australia)', 'Black-and-white_films', 'Brazilian_footballers', 'British_films', 'Columbia_University_alumni', 'Deaths_from_myocardial_infarction', 'English_cricketers', 'English_footballers', 'English-language_albums', 'English-language_films', 'English-language_journals', 'English-language_television_programming', 'Fellows_of_the_Royal_Society', 'French_films', 'German_footballers', 'Guggenheim_Fellows', 'Harvard_University_alumni', 'Hindi-language_films', 'Indian_films', 'Insects_of_Europe', 'Italian_films', 'Italian_footballers', 'Main_Belt_asteroids', 'Major_League_Baseball_pitchers', 'Rivers_of_Romania', 'Russian_footballers', 'Scottish_footballers', 'Serie_A_players', 'The_Football_League_players', 'Villages_in_the_Czech_Republic', 'Villages_in_Turkey', 'Windows_games', 'Yale_University_alumni'])

g_vocab = 100000

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

def getNZFeatureIndices(words, N):
    b_idx = np.array([xxhash.xxh64(w).intdigest()%N for w in words])
    return b_idx

def convert(path, n_vocab, n_classes, classesToIdxMap):
    f =  open(path, 'r')
    lines = f.readlines()
    X = np.zeros((len(lines), n_vocab))
    Y = np.zeros((len(lines), n_classes))
    i = -1
    for l in lines:
        i += 1
        labels, words = getLabelsAndWords(l)
        if (len(labels) <= 0 or len(words) <= 0):
            continue
        # print('labels = ', labels)
        # print('words = ', words)
        index_b = getNZFeatureIndices(words, n_vocab)
        if len(index_b) <= 0:
            print('index_b = ', index_b, 'words =', words, 'labels =', labels)
        for label in labels:
            Y[i, classesToIdxMap[label]] = 1
            # print('label = ', label, 'k = ', k)
        X[i, index_b] += 1
    f.close()
    # print('X.shape, Y.shape', X.shape, Y.shape)
    return X, Y

def convertToFeatures(l, n_vocab, n_classes, classesToIdxMap):
    X = np.zeros(n_vocab)
    Y = np.zeros(n_classes)
    i = -1
    i += 1
    labels, words = getLabelsAndWords(l)
    if (len(labels) <= 0 or len(words) <= 0):
        return X, Y
    # print('labels = ', labels)
    # print('words = ', words)
    index_b = getNZFeatureIndices(words, n_vocab)
    if len(index_b) <= 0:
        print('index_b = ', index_b, 'words =', words, 'labels =', labels)
        return X, Y
    for label in labels:
        Y[classesToIdxMap[label]] = 1
        # print('label = ', label, 'k = ', k)
    X[index_b] += 1
    # print('X.shape, Y.shape', X.shape, Y.shape)
    return X, Y


# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lr', 0.0001, 'Initial learning rate')
tf.app.flags.DEFINE_integer('steps_to_validate', 100, 'Step to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-seperated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 0, "Whether to adopt Distributed Synchronization Mode, 1: sync, 0:async")

# Hyperparameters
learning_rate = FLAGS.lr
steps_to_validate = FLAGS.steps_to_validate

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

def main(_):
    classesToIdxMap = {}
    for i in range(len(g_classes)):
        classesToIdxMap[g_classes[i]] = i
    assert len(classesToIdxMap) == len(g_classes)

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker":worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
            worker_device = "/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster
        )):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            x = tf.placeholder(tf.float32, [None, g_vocab])
            y_ = tf.placeholder(tf.float32, [None, len(g_classes)])
            # Weights
            W = tf.Variable(tf.zeros((g_vocab, len(g_classes))), dtype=tf.float32)
            # bias
            b = tf.Variable(tf.zeros(len(g_classes)), dtype=tf.float32)
            # Probability
            p = 1/(1 + tf.exp( -1 * tf.add(tf.matmul(x, W), b)))
            # Loss
            loss_op = tf.reduce_mean(-1*(tf.multiply(y_, tf.log(p)) + tf.multiply((1-y_), tf.log(1-p))))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss_op)

# DropStaleGradientOptimizer
            if issync == 1:
                # Update gradients in Synchronization Mode
                rep_op = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=len(worker_hosts), replica_id=FLAGS.task_index, total_num_replicas=len(worker_hosts), use_locking=True)
                train_op = rep_op.apply_gradients(grads_and_vars, global_step=global_step)
                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()
            else:
                # Update gradients in Asynchronization Mode
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                saver = tf.train.Saver()
                # tf.summary.scalar('Weight', weight_summary)
                # tf.summary.scalar('Biase', biase_summary)

                # tf.summary.scalar('cost', loss_value)
                # summary_op = tf.summary.merge_all()

                init_op = tf.global_variables_initializer()

            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index==0), logdir="./checkpoint/", init_op=init_op, summary_op=None, saver=saver, global_step=global_step, save_model_secs=60)

            with sv.prepare_or_wait_for_session(server.target) as sess:
                # If is Synchronization Mode
                if FLAGS.task_index == 0 and issync == 1:
                    sv.start_queue_runners(sess, [chief_queue_runner])
                    sess.run(init_token_op)
                step = 0
                # while step < 1000000:
                i = 0
                for line in sys.stdin:
                    i += 1
                    X_train, Y_train = convertToFeatures(line, g_vocab, len(g_classes), classesToIdxMap)
                    _, loss_v, step = sess.run([train_op, loss_op, global_step], feed_dict={x:[X_train], y_:[Y_train]})
                    if step % steps_to_validate == 0:
                        weights, biases = sess.run([W, b])
                        # print('step, weight, biase, loss:', step, weights, biases, loss_v)
                        print('step, loss:', step, loss_v)
                        # sv.summary_computed(sess, summary)
            sv.stop()
if __name__ == "__main__":
    tf.app.run()
