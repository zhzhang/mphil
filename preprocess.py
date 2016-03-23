#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import os
import re
import string
import time

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

def pickle_corpus(path, out):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            os.path.join(root, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess corpus.")
    parser.add_argument('path', type=str, help='path to corpus')
    parser.add_argument('out', type=str, help='path to output dir')
    args = parser.parse_args()
    pickle_corpus(args.path, args.out)

