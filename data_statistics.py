import argparse
import gzip
import numpy
import os
import time

def time_call(f):
    def wrapper(*args):
        time1 = time.time()
        output = f(*args)
        time2 = time.time()
        print "Call to %s completed in %0.3f seconds" % (f.func_name, (time2 - time1))
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
    else:
        return True

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
def process_corpus(path):
    wordmap = get_wordmap(path, 10000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess corpus.")
    parser.add_argument('path', type=str, nargs=1, help='path to corpus')
    args = parser.parse_args()
    process_corpus(args.path[0])
