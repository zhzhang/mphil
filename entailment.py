import argparse
import cPickle as pickle
import time
from dmatrices import DMatrices

def process_data(path, matrices, num_processes, output_path, dense):
    pairs = []
    words = set()
    with open(path, 'r') as f:
        for line in f:
            a,b = line.rstrip('\n').split(' ')
            pairs.append((a,b))
            words.add(a)
            words.add(b)
    dm = DMatrices(matrices, dense=dense)
    t = time.time()
    results = dm.repres(pairs, num_processes=num_processes)
    print "Processed pairs in %d seconds" % (time.time() - t)
    if output_path:
        f = open(output_path, 'w')
    for i, pair in enumerate(pairs):
        if results[i]:
            output_str = "%s %s %0.5f %0.5f" % (pair + results[i])
        else:
            output_str = "%s %s" % (pair)
        if output_path:
            f.write(output_str + "\n")
        else:
            print output_str



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on entailment data set.")
    parser.add_argument('path', type=str, help='path to entailment data data')
    parser.add_argument('matrices', type=str, help='path to matrices')
    parser.add_argument('--num_processes', type=int, help='number of processes to use', default=1)
    parser.add_argument('--output', type=str, help='path to results output path')
    parser.add_argument('--dense', action='store_true', help='flag for dense matrices input')
    args = parser.parse_args()
    process_data(args.path, args.matrices, args.num_processes, args.output, args.dense)

