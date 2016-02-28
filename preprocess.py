#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import re
import string

STOPWORDS = set()
with open('stopwords.txt', 'r') as f:
    for line in f:
        STOPWORDS.add(line.rstrip('\n'))

REMOVE = re.compile('[%s\n]' % re.escape(string.punctuation))

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

