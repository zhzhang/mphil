#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cPickle 
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
def generate_matrices(path, wordmap):
    matrices = {}
    for root, dirnames, filenames in os.walk(path):
        for i, filename in enumerate(filenames):
            print "Processing %s file %d out of %d" % (root, i+1, len(filenames))
            f = open_file(os.path.join(root, filename))
            for line in f:
                tokens = tokenize(line)
                sentence = []
                for token in tokens:
                    if token in wordmap:
                        sentence.append(wordmap[token])
                pairs = get_pairs(sentence)
                for target in tokens:
                    if not target in matrices:
                        matrices[target] = {}
                    for pair in pairs:
                        if pair in matrices[target]:
                            matrices[target][pair] += 1
                        else:
                            matrices[target][pair] = 1
    return matrices

def get_pairs(sentence):
    pairs = []
    for i in range(len(sentence)):
        for j in range(i+1, len(sentence)):
            pairs.append((min(sentence[i], sentence[j]),\
                max(sentence[i], sentence[j])))
    return pairs

def process_corpus(path, wordmap_path):
    if wordmap_path == None:
        wordmap = get_wordmap(path, 2000)
        with open('wordmap.pkl', 'w+') as f:
            cPickle.dump(wordmap, f)
    else:
        with open(wordmap_path, 'r') as f:
            wordmap = cPickle.load(f)
    matrices = generate_matrices(path, wordmap)
    with open('matrices.pkl', 'w+') as f:
        cPickle.dump(matrices, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct density matrices.")
    parser.add_argument('path', type=str, help='path to corpus')
    parser.add_argument('--wordmap', type=str, help='path to wordmap')
    args = parser.parse_args()
    process_corpus(args.path, args.wordmap)

