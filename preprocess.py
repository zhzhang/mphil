#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import re
import string

STOPWORDS = set()
with open('stopwords.txt', 'r') as f:
    for line in f:
        STOPWORDS.add(line.rstrip('\n'))

PUNC = re.compile('[%s]' % re.escape(string.punctuation))

def open_file(path):
    if path.endswith('.gz'):
        f = gzip.open(path, 'r')
    else:
        f = open(path, 'r')
    return f

def tokenize(line):
    return map(lambda x: x.lower(), filter(filter_helper, line.split(' ')))

def filter_helper(word):
    tmp = PUNC.sub('', word)
    if '<doc' in word or 'doc>' in word:
        return False
    elif word in STOPWORDS:
        return False
    elif '.\n' in word:
        return False
    elif len(tmp) == 0:
        return False
    else:
        return True

