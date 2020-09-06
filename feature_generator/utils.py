import pickle

def save_datafile(path, name, data):
    with open(path.joinpath(name), "wb") as f:
        pickle.dump(data, f, protocol=2)
