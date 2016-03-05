#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cPickle 
import numpy as np
import os
import time
from multiprocessing import Pool, Manager
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
def get_wordmap(path, threshold):
    wordcount = {}
    for root, dirnames, filenames in os.walk(path):
        for i, filename in enumerate(filenames):
            print "Processing %s file %d out of %d" % (root, i+1, len(filenames))
            f = open_file(os.path.join(root, filename))
            for line in f:
                tokens = tokenize(line)
                for token in tokens:
                    if not token in wordcount:
                        wordcount[token] = 1
                    else:
                        wordcount[token] += 1
    words = map(lambda x: (x, wordcount[x]), wordcount.keys())
    words.sort(key=lambda x: x[1], reverse=True)
    wordmap = {}
    index = 0
    while index < threshold and index < len(words):
        wordmap[words[index][0]] = index
        index += 1
    return wordmap

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
    coverages = map(lambda x: len(nonzeros[x]), nonzeros.keys())
    print float(sum(coverages)) / len(coverages)

@time_call
def generate_matrices(path, wordmap, cores, targets):
    args = []
    manager = Manager()
    lock = manager.Lock()
    matrices = manager.dict()
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            args.append((root, filename, matrices, lock, wordmap, targets))
    pool = Pool(processes=cores if not cores == None else 1)
    pool.map(generate_matrices_worker, args)
    return dict(matrices) # TODO: Find out if there's anything wrong with this.

def generate_matrices_worker(args):
    root, filename, total_matrices, lock, wordmap, targets = args
    print "Processing %s in %s" % (filename, root)
    f = open_file(os.path.join(root, filename))
    matrices = {}
    for line in f:
        tokens = tokenize(line)
        sentence = []
        for token in tokens:
            if token in wordmap:
                sentence.append(wordmap[token])
        context = get_context(sentence)
        for target in tokens:
            if not targets == None and not target in targets:
                continue
            if not target in matrices:
                matrices[target] = {}
            keys = context.keys()
            tid = wordmap[target] if target in wordmap else -1
            for i in range(len(keys)):
                for j in range(i, len(keys)):
                    a = min(keys[i], keys[j])
                    b = max(keys[i], keys[j])
                    count_a = context[a]
                    count_b = context[b]
                    if tid == a:
                        count_a -= 1
                    if tid == b:
                        count_b -= 1
                    if count_a > 0 and count_b > 0:
                        matrices[target][(a,b)] = count_a * count_b
    # Update the total matrix
    lock.acquire()
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
    lock.release()

def get_context(sentence):
    word_counts = {}
    for word in sentence:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts

def verify_matrix(output, correct):
    pass

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

def process_corpus(path, wordmap_path, cores, targets):
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
    matrices = generate_matrices(path, wordmap, cores, targets)
    print_matrices(matrices, wordmap)
    with open('matrices.pkl', 'w+') as f:
        cPickle.dump(matrices, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct density matrices.")
    parser.add_argument('path', type=str, help='path to corpus')
    parser.add_argument('--targets', type=str, help='path to pickled target dict')
    parser.add_argument('--wordmap', type=str, help='path to wordmap')
    parser.add_argument('--cores', type=int, help='number of cores to use')
    args = parser.parse_args()
    process_corpus(args.path, args.wordmap, args.cores, args.targets)

