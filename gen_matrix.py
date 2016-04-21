#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cPickle 
import numpy as np
import os
import time
from multiprocessing import Pool
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
def check_sparsity(path, wordmap):
    nonzeros = {}
    for root, dirnames, filenames in os.walk(path):
        for i, filename in enumerate(filenames):
            f = open_file(os.path.join(root, filename))
            for line in f:
                tokens = tokenize(line)
                sentence = []
                for token in tokens:
                    if token in wordmap:
                        sentence.append(wordmap[token])
                sentence = set(sentence)
                for target in tokens:
                    if not target in nonzeros:
                        nonzeros[target] = sentence
                    else:
                        nonzeros[target] = nonzeros[target].union(sentence)
            f.close()
    coverages = map(lambda x: len(nonzeros[x]), nonzeros.keys())
    print float(sum(coverages)) / len(coverages)

@time_call
def generate_matrices(path, wordmap, cores, targets, output_dir):
    args = []
    pool = Pool(processes=cores if not cores == None else 1)
    output_dir = os.path.join(output_dir, 'matrices')
    for target in targets:
        args.append((target, path, output_dir, wordmap))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pool.map(generate_matrices_worker, args)
    pool.close()
    pool.join()

def generate_matrices_worker(args):
    target, path, output_dir, wordmap = args
    print "Generating density matrix for '%s'" % target
    matrix = {}
    tid = wordmap[target]
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            f = open_file(os.path.join(root, filename))
            for line in f:
                pruned = False # Remove the first instance of the target word.
                target_count = 0
                for token in line:
                    if token == tid:
                        target_count += 1
                if not target_count > 0:
                    continue
                context = get_context(sentence, wordmap['_d'])
                if tid < wordmap['_d']:
                    if context[tid] == 1:
                        del context[tid]
                    else:
                        context[tid] = context[tid] - 1
                keys = context.keys()
                for i in range(len(keys)):
                    for j in range(i, len(keys)):
                        a = min(keys[i], keys[j])
                        b = max(keys[i], keys[j])
                        count_a = context[a]
                        count_b = context[b]
                        if (a,b) in matrix:
                            matrix[(a,b)] +=\
                              count_a * count_b * target_count
                        else:
                            matrix[(a,b)] =\
                              count_a * count_b * target_count
    # Save matrix.
    with open(os.path.join(output_dir, target + '.pkl'), 'w') as f:
        pickle.dump(matrix, f)

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

def print_matrices(matrices, wordmap):
    reverse_wordmap = {}
    for word in wordmap:
        reverse_wordmap[wordmap[word]] = word
    for target in matrices:
        print "-----"
        print "Target: %s" % target
        for pair in matrices[target]:
            print reverse_wordmap[pair[0]], reverse_wordmap[pair[1]],\
              matrices[target][pair]

def process_corpus(path, wordmap_path, cores, targets, verbose, output_dir):
    if wordmap_path == None:
        wordmap = get_wordmap(path, 2000)
        with open('wordmap.pkl', 'w+') as f:
            cPickle.dump(wordmap, f)
    else:
        with open(wordmap_path, 'r') as f:
            wordmap = cPickle.load(f)
    if not targets == None:
        with open(targets, 'r') as f:
            target_pairs = cPickle.load(f)
        targets = set()
        for a, b in target_pairs:
            targets.add(a)
            targets.add(b)
    generate_matrices(path, wordmap, cores, targets, output_dir)
    if verbose:
        print_matrices(matrices, wordmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct density matrices.")
    parser.add_argument('path', type=str, help='path to corpus')
    parser.add_argument('output', type=str, help='path to output directory')
    parser.add_argument('--targets', type=str, help='path to pickled target dict')
    parser.add_argument('--wordmap', type=str, help='path to wordmap')
    parser.add_argument('--cores', type=int, help='number of cores to use')
    parser.add_argument('-v', help='verbose', action='store_true')
    args = parser.parse_args()
    process_corpus(args.path, args.wordmap, args.cores, args.targets,\
      args.v, args.output)

