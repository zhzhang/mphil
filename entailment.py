import argparse
import cPickle as pickle
import numpy as np
import os
import time
from dmatrices import DMatrices, ZERO_THRESH

def process_data(path, matrices_path, num_processes, output_path, dense):
    with open(path, 'r') as f:
        data = pickle.load(f)
    dm = DMatrices(matrices_path, dense=dense)
    t = time.time()
    results = dm.repres(data.keys(), num_processes=num_processes)
    print "Representativeness computed in %d seconds" % (time.time() - t)
    evaluate(data, results)
    if output_path:
        f = open(output_path, 'w')
    for i, pair in enumerate(data):
        if results[i]:
            output_str = "%s %s %0.5f %0.5f" % (pair + results[i])
        else:
            output_str = "%s %s" % (pair)
        if output_path:
            f.write(output_str + "\n")
    vectors_path = os.path.join(matrices_path, "vectors.txt")
    if os.path.exists(vectors_path):
        vector_results = process_vectors(data.keys(), vectors_path)
        evaluate(data, vector_results)

def process_vectors(pairs, vectors_path):
    vectors = {}
    with open(vectors_path, 'r') as f:
        for line in f:
            data = line.rstrip('\n').split(' ')
            vectors[data[0]] = np.array([float(x) for x in data[1:]])
    results = []
    for pair in pairs:
        if not (pair[0] in vectors and pair[1] in vectors):
            results.append(None)
            continue
        vecx = vectors[pair[0]]
        vecy = vectors[pair[1]]
        x_entails_y = 1
        y_entails_x = 1 
        for i in range(len(vecx)):
            if vecx[i] > ZERO_THRESH and vecy[i] <= ZERO_THRESH:
                x_entails_y = 0
            if vecy[i] > ZERO_THRESH and vecx[i] <= ZERO_THRESH:
                y_entails_x = 0 
        results.append((x_entails_y, y_entails_x))
    return results

def evaluate(ground_truth, results):
    # Evaluated by partial correctness.
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    # Strictly correct.
    correct = 0
    # Total data points where matrices exist.
    total = 0
    # Counts for each data type.
    pos = 0
    neg = 0
    for i, pair in enumerate(ground_truth):
        if results[i] == None:
            continue
        total += 1
        r_ab, r_ba = results[i]
        if ground_truth[pair][0] == "hyper":
            pos += 1
            if r_ab > r_ba:
                true_pos += 1
                if r_ba < ZERO_THRESH:
                    correct += 1
            elif r_ab <= r_ba:
                false_neg += 1
        if ground_truth[pair][0] == "random_n":
            neg += 1
            if r_ab > r_ba:
                false_pos += 1
            elif r_ab <= r_ba:
                true_neg += 1
    print "Total pairs with complete data: %d out of %d" % (total, len(ground_truth))
    print "Completely correct %d out of %d, %0.1f%%" % (correct, pos, 100 * correct / float(pos))
    if pos > 0:
        print "True-Pos: %d = %0.1f%% out of %d" % (true_pos, 100 * true_pos / float(pos), pos)
    if neg > 0:
        print "True-Neg: %d = %0.1f%% out of %d" % (true_neg, 100 * true_neg / float(neg), neg)
    print "F1 score"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on entailment data set.")
    parser.add_argument('path', type=str, help='path to entailment data')
    parser.add_argument('matrices', type=str, help='path to matrices')
    parser.add_argument('--num_processes', type=int, help='number of processes to use', default=1)
    parser.add_argument('--output', type=str, help='path to results output path')
    parser.add_argument('--dense', action='store_true', help='flag for dense matrices input')
    args = parser.parse_args()
    process_data(args.path, args.matrices, args.num_processes, args.output, args.dense)

