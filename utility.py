from sklearn.model_selection import StratifiedShuffleSplit


class Data:
    WIDTH = 32
    HEIGHT = 32
    CHANNEL = 3


def loadPickle(filepath):
    import cPickle
    with open(filepath, 'rb') as fi:
        dict = cPickle.load(fi)
    return dict


def writePickle(var, filepath):
    import cPickle
    with open(filepath, 'wb') as fo:
        cPickle.dump(var, fo)


def stratifiedSplitTrainSet(trn_encoded, labels, num_splits):
    # Split data to train set and dev set
    split = StratifiedShuffleSplit(
        n_splits=num_splits, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(trn_encoded, labels):
        trn_data = trn_encoded[train_index]
        trn_labels = labels[train_index]

        dev_data = trn_encoded[test_index]
        dev_labels = labels[test_index]

    return trn_data, trn_labels, dev_data, dev_labels
