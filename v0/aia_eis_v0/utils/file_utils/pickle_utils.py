import os
import pickle

def pickle_file(obj, fn, fp='./'):
    with open(os.path.join(fp, fn), 'wb') as file:
        pickle.dump(obj, file)

def load_pickle_file(fp, fn):
    with open(os.path.join(fp, fn), 'rb') as file:
        obj = pickle.load(file)
    return obj