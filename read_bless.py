#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cPickle 
import os
from preprocess import *

def parse_line(line):
    # Concept, class, relation, relatum
    (con, cla, rel, relu) = line.split('\t')
    con = con.split('-')[0]
    relu = relu.split('-')[0]
    return (con, relu, cla, rel)

def process_bless(path, output_path, subsample=1):
    output = {}
    count = 0
    with open(path, 'r') as f:
        for line in f:
            con, relu, cla, rel = parse_line(line)
            if rel == 'hyper' or rel == 'coord' or rel == 'mero':
                output[(con, relu)] = (rel, cla)
            elif rel == 'random-n':
                count += 1
                if count == subsample:
                    output[(con, relu)] = (rel, cla)
                    count = 0
    if output_path:
        print "Saving output..."
        with open(output_path + '.txt', 'w+') as f:
            for pair in output:
                f.write("%s %s\n" % pair)
        with open(output_path + '.pkl', 'w+') as f:
            pickle.dump(output, f)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read in BLESS data set.")
    parser.add_argument('path', type=str, help='path to BLESS data set')
    parser.add_argument('--output', type=str, help='output file')
    args = parser.parse_args()
    process_bless(args.path, args.output, subsample=4)

