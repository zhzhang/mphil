import argparse
import cPickle as pickle
import time
from dmatrices import DMatrices

def process_bless(path, matrices, wordmap, num_cores):
    with open(path, 'r') as f:
        bless = pickle.load(f)
    print len(bless)
    dm = DMatrices(matrices, wordmap)
    words = set()
    for a,b in bless:
        words.add(a)
        words.add(b)
    t = time.time()
    dm.get_eigenvectors(words, num_cores=num_cores)
    print "Eigenvectors computed in %d seconds" % (time.time() - t)
    t = time.time()
    dm.repres(bless, num_cores=num_cores)
    print "Processed pairs in %d seconds" % (time.time() - t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on BLESS.")
    parser.add_argument('path', type=str, help='path to preprocessed BLESS data')
    parser.add_argument('matrices', type=str, help='path to pickled matrices')
    parser.add_argument('wordmap', type=str, help='path to pickled wordmap')
    parser.add_argument('--num_cores', type=int, help='number of cores to use', default=1)
    args = parser.parse_args()
    process_bless(args.path, args.matrices, args.wordmap, args.num_cores)

