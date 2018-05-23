import cv2
import numpy as np
from utility import loadPickle, writePickle
from utility import Data
from sklearn.cluster import KMeans


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


def createBagOfWords():
    descs1 = np.array(loadPickle('./descriptors/descs_1'))
    descs2 = np.array(loadPickle('./descriptors/descs_2'))
    descs3 = np.array(loadPickle('./descriptors/descs_3'))
    descs4 = np.array(loadPickle('./descriptors/descs_4'))
    descs5 = np.array(loadPickle('./descriptors/descs_5'))

    X = np.concatenate((descs1, descs2, descs3, descs4, descs5), axis=0)

    kmeans = KMeans(n_clusters=500, random_state=42)
    kmeans.fit(X)

    return kmeans


if __name__ == '__main__':
    # writeDescriptors()
    kmeans = loadPickle('./bag-of-words/bow_500')
    print kmeans.cluster_centers_
