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
    weeds = dm.weeds_prec(data.keys(), num_processes=num_processes)
    print "AP WeedsPrec: %0.3f" % get_avg_precision(data, weeds)
    clarke_de = dm.clarke_de(data.keys(), num_processes=num_processes)
    print "AP ClarkeDE: %0.3f" % get_avg_precision(data, clarke_de)
    t = time.time()
    results = dm.repres(data.keys(), num_processes=num_processes)
    print "Representativeness computed in %d seconds" % (time.time() - t)
    evaluate(data, results)
    if output_path:
        with open(output_path, 'w') as f:
            for i, pair in enumerate(data):
                if results[i]:
                    output_str = "%s %s %0.6f %0.6f" % (pair + results[i])
                else:
                    output_str = "%s %s" % (pair)
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
    nonzero = 0
    for i, pair in enumerate(ground_truth):
        if results[i] == None:
            continue
        total += 1
        r_ab, r_ba = results[i]
        if r_ab >= ZERO_THRESH or r_ba >= ZERO_THRESH:
            nonzero += 1
        if ground_truth[pair][0] == "hyper":
            pos += 1
            if r_ab > r_ba and r_ab >= ZERO_THRESH:
                true_pos += 1
                if r_ba < ZERO_THRESH:
                    correct += 1
            elif r_ab <= r_ba:
                false_neg += 1
        elif ground_truth[pair][0] == "random-n":
            neg += 1
            if r_ab > r_ba and r_ab >= ZERO_THRESH:
                false_pos += 1
            elif r_ab <= r_ba:
                true_neg += 1
    print "Total pairs with complete data: %d out of %d" % (total, len(ground_truth))
    print "Total pairs with nonzero data: %d out of %d" % (nonzero, len(ground_truth))
    print "Completely correct %d out of %d, %0.1f%%" % (correct, pos, 100 * correct / float(pos))
    print "TP: %d  FP: %d  TN: %d  FN: %d" % (true_pos, false_pos, true_neg, false_neg)
    print "...out of POS: %d  NEG: %d" % (pos, neg)

def get_avg_precision(ground_truth, results):
    concept_result_map = {}
    for i, pair in enumerate(ground_truth):
        if pair[0] in concept_result_map:
            concept_result_map[pair[0]].append((results[i], ground_truth[pair][0]))
        else:
            concept_result_map[pair[0]] = [(results[i], ground_truth[pair][0])]
    final = 0.0
    total_concepts = 0
    for concept in concept_result_map:
        val_label_pairs = concept_result_map[concept]
        val_label_pairs = sorted(val_label_pairs, reverse=True)
        correct = 0
        total = 0.0
        for k, (val, label) in enumerate(val_label_pairs):
            if label == "hyper":
                correct += 1
                total += correct / float(k+1)
            elif val == None:
                break
        if k > 0:
            total_concepts += 1
            final += total / k
    return final / total_concepts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on entailment data set.")
    parser.add_argument('path', type=str, help='path to entailment data')
    parser.add_argument('matrices', type=str, help='path to matrices')
    parser.add_argument('--num_processes', type=int, help='number of processes to use', default=1)
    parser.add_argument('--output', type=str, help='path to results output path')
    parser.add_argument('--dense', action='store_true', help='flag for dense matrices input')
    args = parser.parse_args()
    process_data(args.path, args.matrices, args.num_processes, args.output, args.dense)

