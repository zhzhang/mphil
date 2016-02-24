import argparse
import numpy

def open_file(path):
    f = open(path, 'r')
    return f

def tokenize(line):
    return map(lambda x: x.lower(), filter(filter_helper, line))

def filter_helper(word):
    if '<doc' in word or 'doc>' in word:
        return True 
    else:
        return False

def get_wordmap(path):
    f = open_file(path)
    wordmap = {}
    index = 0
    for line in f:
        tokens = tokenize(line)
        for token in tokens:
            if not token in wordmap:
                wordmap[token] = index
                index += 1
    return wordmap

def process_corpus(path):
    wordmap = get_wordmap(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess corpus.")
    parser.add_argument('path', type=str, nargs=1, help='path to corpus')
    args = parser.parse_args()
    process_corpus(args.path[0])
