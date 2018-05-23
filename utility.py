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
