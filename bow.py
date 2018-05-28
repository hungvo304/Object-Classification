import cv2
import numpy as np
from utility import loadPickle, writePickle
from utility import Data
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree


def extractSIFTFeature(grayscaleImage):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(grayscaleImage, None)
    return keypoints, descriptors


def data2image(data_batch_row):
    return np.transpose(data_batch_row.reshape(
        Data.CHANNEL, Data.HEIGHT, Data.WIDTH), (1, 2, 0))


def getSiftDescriptorsFromData(data_batch):
    descsList = None
    for row in data_batch['data']:
        img = data2image(row)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        descs = extractSIFTFeature(gray)[1]
        if descs is None:
            continue
        if descsList is None:
            descsList = descs
        else:
            descsList = np.concatenate((descsList, descs), axis=0)

    return descsList


def writeDescriptors():
    '''write all descriptors variables to files'''

    paths = ['./cifar-10-batches-py/data_batch_1',
             './cifar-10-batches-py/data_batch_2',
             './cifar-10-batches-py/data_batch_3',
             './cifar-10-batches-py/data_batch_4',
             './cifar-10-batches-py/data_batch_5']
    for filepath in paths:
        trn = loadPickle(filepath)
        descsList = getSiftDescriptorsFromData(trn)
        writePickle(descsList, './descriptors/desc_' +
                    filepath.split('/')[-1].split('_')[-1])


def createBagOfWords(num_vocabs=500):
    descs1 = np.array(loadPickle('./descriptors/descs_1'))
    descs2 = np.array(loadPickle('./descriptors/descs_2'))
    descs3 = np.array(loadPickle('./descriptors/descs_3'))
    descs4 = np.array(loadPickle('./descriptors/descs_4'))
    descs5 = np.array(loadPickle('./descriptors/descs_5'))

    X = np.concatenate((descs1, descs2, descs3, descs4, descs5), axis=0)

    kmeans = KMeans(n_clusters=num_vocabs, random_state=42)
    kmeans.fit(X)

    return kmeans


def encodeImage(img, bow):
    # extract sift features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, descs = extractSIFTFeature(gray)

    # create histogram of visual words
    histOfVW = np.zeros((500))
    if descs is None:
        return histOfVW

    # increase bins of historgram
    tree = KDTree(bow)
    for desc in descs:
        _, idx = tree.query([desc], k=1)
        histOfVW[idx] += 1

    return histOfVW


def encodeBatch(data_batch, bow):
    batch_encoded = []
    for row in data_batch['data']:
        img = data2image(row)
        batch_encoded.append(encodeImage(img, bow))

    return batch_encoded


if __name__ == '__main__':
    kmeans = createBagOfWords(num_vocabs=1000)
    writePickle(kmeans, './bag-of-words/bow_1000')
