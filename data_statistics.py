import argparse
import gzip
import numpy
import os

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
        return True 
    else:
        return False

def get_wordmap(path):
    wordmap = {}
    index = 0
    for root, dirnames, filenames in os.walk(path):
        for i, filename in enumerate(filenames):
            print "Processing %s file %d out of %d" % (root, i+1, len(filenames))
            f = open_file(os.path.join(root, filename))
            for line in f:
                tokens = tokenize(line)
                for token in tokens:
                    if not token in wordmap:
                        wordmap[token] = index
                        index += 1
    return wordmap

def process_corpus(path):
    wordmap = get_wordmap(path)
    print len(wordmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess corpus.")
    parser.add_argument('path', type=str, nargs=1, help='path to corpus')
    args = parser.parse_args()
    process_corpus(args.path[0])
