#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cPickle as pickle
import numpy as np
import os
import time
from preprocess import *

def time_call(f):
    def wrapper(*args):
        time1 = time.time()
        output = f(*args)
        time2 = time.time()
        print "Call to %s completed in %0.3f seconds" % (f.func_name, (time2 - time1))
        return output
    return wrapper

@time_call
def generate_matrices(path, cores, targets, wordmap):
    args = []
    matrices = {}
    for target in targets:
        matrices[target] = {}
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            process_file(root, filename, matrices, targets, wordmap['_d'])
    return matrices

def process_file(root, filename, total_matrices, targets, dim):
    print "Processing %s in %s" % (filename, root)
    time1 = time.time()
    f = open_file(os.path.join(root, filename))
    f = pickle.load(f)
    matrices = {}
    for line in f:
        context = get_context(line, dim)
        # Get normalization constant for vectors.
        normalizer = 0.0
        for x in context:
            normalizer += context[x] ** 2
        for target in line:
            if not targets == None and not target in targets:
                continue
            if not target in matrices:
                matrices[target] = {}
            keys = context.keys()
            if target in context:
                final_normalizer = normalizer - context[target] ** 2\
                  + (context[target] - 1) ** 2
            else:
                final_normalizer = normalizer
            for i in range(len(keys)):
                for j in range(i, len(keys)):
                    a = min(keys[i], keys[j])
                    b = max(keys[i], keys[j])
                    count_a = context[a]
                    count_b = context[b]
                    if target == a:
                        count_a -= 1
                    if target == b:
                        count_b -= 1
                    if count_a > 0 and count_b > 0:
                        if (a,b) in matrices[target]:
                            matrices[target][(a,b)] +=\
                              count_a * count_b / final_normalizer
                        else:
                            matrices[target][(a,b)] =\
                              count_a * count_b / final_normalizer
    time2 = time.time()
    print "Processing %s in %s completed in %0.3f seconds" % (filename, root, (time2 - time1))
    # Update the total matrix
    time1 = time.time()
    for target in matrices:
        target_matrix = matrices[target]
        if target in total_matrices:
            total_target_matrix = total_matrices[target]
        else:
            total_target_matrix = {}
        for token in target_matrix:
            if token in total_target_matrix:
                total_target_matrix[token] += target_matrix[token]
            else:
                total_target_matrix[token] = target_matrix[token]
        total_matrices[target] = total_target_matrix
    time2 = time.time()
    print "Merging %s in %s completed in %0.3f seconds" % (filename, root, (time2 - time1))

def get_context(sentence, dim):
    word_counts = {}
    for word in sentence:
        if word >= dim:
            continue
        elif word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts

def process_corpus(path, wordmap_path, cores, targets, verbose, out='matrices.pkl'):
    if not targets == None:
        with open(wordmap_path, 'r') as f:
            wordmap = pickle.load(f)
        with open(targets, 'r') as f:
            target_pairs = pickle.load(f)
        targets = set()
        for a, b in target_pairs:
            targets.add(wordmap[a])
            targets.add(wordmap[b])
    matrices = generate_matrices(path, cores, targets, wordmap)
    if verbose:
        print_matrices(matrices, wordmap)
    with open(out, 'w+') as f:
        pickle.dump(matrices, f)

def get_statistics(path, wordmap, dim):
    if wordmap == None:
        wordmap = get_wordmap(path, dim, None)
    else:
        with open(wordmap, 'r') as f:
            wordmap = pickle.load(f)
    get_unique_contexts(path, wordmap, dim)

def get_unique_contexts(path, wordmap, dim):
    contexts = {}
    total = 0
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            print "Processing %s ..." % path
            f = open_file(path)
            for line in f:
                context = get_context(line, dim)
                c = tuple([(x, context[x]) for x in sorted(context)])
                if c in contexts:
                    contexts[c] += 1
                else:
                    contexts[c] = 1
    with open('unique_contexts.pkl', 'w+') as f:
        pickle.dump((contexts,total), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct density matrices.")
    parser.add_argument('path', type=str, help='path to corpus')
    parser.add_argument('--out', type=str, help='path to output dir')
    parser.add_argument('--wordmap', type=str, help='path to wordmap')
    parser.add_argument('--targets', type=str, help='path to pickled target dict')
    parser.add_argument('--cores', type=int, help='number of cores to use')
    parser.add_argument('-v', help='verbose', action='store_true')
    parser.add_argument('-s', action='store_true',\
      help='flag to retrieve corpus statistics instead of preprocessing the corpus')
    args = parser.parse_args()
    if not args.out == None:
        process_corpus(args.path, args.wordmap, args.cores, args.targets,\
          args.v)
    elif args.s:
        get_statistics(args.path, args.wordmap, args.dim)

