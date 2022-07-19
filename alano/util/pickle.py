import pickle

def save_obj(obj, filename):
    if filename.endswith('.pkl'):
        filename = filename[:-4]
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    if filename.endswith('.pkl'):
        filename = filename[:-4]
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)
