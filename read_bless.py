#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cPickle 
import os
from preprocess import *

def parse_line(line):
    # Concept, class, relation, relatum
    (con, cla , rel, relu) = line.split('\t')
    con = con.split('-')[0]
    return (con, cla)

def process_bless(path, p):
    output = set()
    with open(path, 'r') as f:
        for line in f:
            data = parse_line(line)
            output.add(data)
    if p:
        print "Pickling..."
        with open('bless.pkl', 'w+') as f:
            cPickle.dump(output, f)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read in BLESS data set.")
    parser.add_argument('path', type=str, help='path to BLESS data set')
    parser.add_argument('-p', help='pickle the output', action='store_true')
    args = parser.parse_args()
    process_bless(args.path, args.p)

