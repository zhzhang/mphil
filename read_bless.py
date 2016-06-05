#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import cPickle as pickle
import math
import os
import random

plant = ['tree', 'fruit', 'vegetable']
nonliving = ['building', 'weapon', 'container', 'musical_instrument', 'appliance', 'tool', 'vehicle', 'clothing', 'furniture']
animal = ['insect', 'water_animal', 'amphibian_reptile', 'bird', 'ground_mammal']

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
    for con, relu, cla, rel in data:
        concepts.add(con)
    sample = random.sample(concepts, int(math.floor(len(concepts) / 4.7)))
    output = []
    for con, relu, cla, rel in data:
        if con in sample:
            output.append((con, relu, cla, rel))
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
            if not get_group(classmap[point[0]]) == get_group(cla):
                output.append((con, point[1], cla, 'random-n'))
                datacopy.remove(point)
    return output

def get_group(cla):
    if cla in nonliving:
        return "nonliving"
    elif cla in animal:
        return "animal"
    elif cla in plant:
        return "plant"

if __name__ == "__main__":
    if not os.path.exists("targets"):
        os.mkdir("targets")
    nouns = []
    with open("bless.txt", 'r') as f:
        for line in f:
            con, relu, cla, rel = parse_line(line)
            if rel == 'hyper' or rel == 'coord' or rel == 'mero' or rel == 'random-n':
                nouns.append((con, relu, cla, rel))
    write_data("nouns", nouns)

    hyper = []
    randomn = []
    with open("bless_random_test.txt", 'r') as f:
        for line in f:
            con, relu, cla, rel = parse_line(line)
            if rel == 'hyper':
                hyper.append((con, relu, cla, rel))
            elif rel == 'random-n':
                randomn.append((con, relu, cla, rel))
    write_data("rand-hyper", hyper + subsample(randomn))
    write_data("hyper", hyper) 
    write_data("shuffle-hyper", hyper + shuffle(hyper)) 

