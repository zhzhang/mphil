#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import cPickle
import math
import os
import random
from preprocess import *

def parse_line(line):
    # Concept, class, relation, relatum
    (con, cla, rel, relu) = line.split('\t')
    con = con.split('-')[0]
    relu = relu.split('-')[0]
    return (con, relu, cla, rel)

def write_data(name, data):
    with open("targets/" + name + '.txt', 'w') as f:
        for con, relu, _, _ in data:
            f.write("%s %s\n" % (con, relu))
    with open("targets/" + name + '.pkl', 'w') as f:
        pickle.dump(data, f)

def subsample(data):
    concepts = set()
    for con, relu, rel, cla in data:
        concepts.add(con)
    sample = random.sample(concepts, math.floor(concepts / 6))
    output = []
    for con, relu, rel, cla in data:
        if con in sample:
            concepts.add(con)
    print "Random samples: %d" % len(output)
    return output

def shuffle(data):
    datacopy = copy.deepcopy(data)
    random.shuffle(datacopy)
    output = []
    classmap = {}
    for con, relu, cla, rel in data:
        classmap[con] = cla
    for con, _, cla, _ in data:
        for point in datacopy:
            if not classmap[point[0]] == cla:
                output.append((con, point[1], cla, 'random-n'))
                datacopy.remove(point)
    return output


if __name__ == "__main__":
    if not os.path.exists("targets")
        os.mkdir("targets")
    nouns = []
    with open("bless.txt", 'r') as f:
        for line in f:
            con, relu, cla, rel = parse_line(line)
            if rel == 'hyper' or rel == 'coord' or rel == 'mero' or rel == 'random-n':
                nouns.append((con, relu, rel, cla))
    write_data("nouns", nouns)

    hyper = []
    random = []
    with open("bless_random_test.txt", 'r') as f:
        for line in f:
            con, relu, cla, rel = parse_line(line)
            if rel == 'hyper':
                hyper.append((con, relu, rel, cla))
            elif rel == 'random-n':
                random.append((con, relu, rel, cla))
    write_data("rand-hyper", hyper + subsample(random))
    write_data("hyper", hyper) 
    write_data("shuffle-hyper", hyper + shuffle(hyper)) 

