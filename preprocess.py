#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cPickle as pickle
import gzip
import os
import re
import string
import time
from multiprocessing import Pool

STOPWORDS = set()
with open('stopwords.txt', 'r') as f:
    for line in f:
        STOPWORDS.add(line.rstrip('\n'))

REMOVE = re.compile('[%s\n]' % re.escape(string.punctuation))

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
    tmp = map(lambda x: REMOVE.sub('', x).lower(), line.split(' '))
    return filter(filter_helper, tmp)

def filter_helper(word):
    if '<doc' in word or 'doc>' in word:
        return False
    elif word in STOPWORDS:
        return False
    elif len(word) == 0:
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

def preprocess_corpus(path, out, cores, wordmap):
    if wordmap == None
        wordmap = get_wordmap(path, 2000)
    else:
        with open(wordmap, 'r') as f:
            wordmap = pickle.load(f)
    args = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            args.append((root, filename, out, wordmap))
    pool = Pool(processes=cores if not cores == None else 1)
    pool.map(process_file, args)
    pool.close()
    pool.join()

def process_file(args):
    root, filename, out, wordmap = args
    output = []
    path = os.path.join(root, filename)
    f = open_file(path)
    for line in f:
        output_line = []
        tokens = tokenize(line)
        for token in tokens:
            if token in wordmap:
                output_line.append(wordmap[token])
        if len(output_line) > 0:
            output.append(output_line)
    newroot = os.path.join(out, *os.path.split(root)[1:])
    if not os.path.isdir(newroot):
        os.makedirs(newroot)
    pre, _ = os.path.splitext(filename)
    with open(os.path.join(newroot, pre + '.pkl'), 'w+') as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess corpus.")
    parser.add_argument('path', type=str, help='path to corpus')
    parser.add_argument('out', type=str, help='path to output dir')
    parser.add_argument('--cores', type=int, help='number of cores to use')
    parser.add_argument('--wordmap', type=str, help='path to wordmap')
    args = parser.parse_args()
    preprocess_corpus(args.path, args.out, args.cores, args.wordmap)

