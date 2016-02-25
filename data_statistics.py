#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cPickle 
import gzip
import numpy as np
import os
import re
import time

STOPWORDS = set()
with open('stopwords.txt', 'r') as f:
    for line in f:
        STOPWORDS.add(line.rstrip('\n'))

def time_call(f):
    def wrapper(*args):
        time1 = time.time()
        output = f(*args)
        time2 = time.time()
        print "Call to %s completed in %0.3f seconds" % (f.func_name, (time2 - time1))
        return output
    return wrapper

def open_file(path):
    if path.endswith('.gz'):
        f = gzip.open(path, 'r')
    else:
        f = open(path, 'r')
    return f

def tokenize(line):
    return map(lambda x: x.lower(), filter(filter_helper, line.split(' ')))

def filter_helper(word):
    if '<doc' in word or 'doc>' in word:
        return False
    elif word in STOPWORDS:
        return False
    elif '.\n' in word:
        return False
    else:
        return True

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
                for target in tokens:
                    if not target in nonzeros:
                        nonzeros[target] = np.zeros(len(wordmap))
                    for token in tokens:
                        if token in wordmap:
                            nonzeros[target][wordmap[token]] = 1
    coverages = map(lambda x: sum(nonzeros[x]), nonzeros.keys())
    print float(sum(coverages)) / len(coverages)

def process_corpus(path, wordmap_path):
    if wordmap_path == None:
        wordmap = get_wordmap(path, 2000)
        with open('wordmap.pkl', 'w+') as f:
            cPickle.dump(wordmap, f)
    else:
        with open(wordmap_path, 'r') as f:
            wordmap = cPickle.load(f)
    check_sparsity(path, wordmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess corpus.")
    parser.add_argument('path', type=str, help='path to corpus')
    parser.add_argument('--wordmap', type=str, help='path to wordmap')
    args = parser.parse_args()
    process_corpus(args.path, args.wordmap)

