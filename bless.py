import argparse
import cPickle as pickle
from dmatrices import DMatrices

def process_bless(path, matrices, wordmap):
    with open(path, 'r') as f:
        bless = pickle.load(f)
    dm = DMatrices(matrices, wordmap)
    for x, y in bless:
        rxy = dm.repres(x,y)
        ryx = dm.repres(y,x)
        print x,y,rxy,ryx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on BLESS.")
    parser.add_argument('path', type=str, help='path to preprocessed BLESS data')
    parser.add_argument('matrices', type=str, help='path to pickled matrices')
    parser.add_argument('wordmap', type=str, help='path to pickled wordmap')
    args = parser.parse_args()
    process_bless(args.path, args.matrices, args.wordmap)

