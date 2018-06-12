import torch
import numpy as np
import cv2
from torchvision import transforms, datasets, models
from utility import writePickle
from PIL import Image
from bow import data2image


def image_loader(img):
    loader = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = Image.open(image_path)
    image = Image.fromarray(cv2_im)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def encodeImage_deepfeat(img):
    image = image_loader(img)

    model = models.alexnet(pretrained=True)
    newclassifier = torch.nn.Sequential(
        *list(model.classifier.children())[:-1])
    model.classifier = newclassifier

    featureVector = model(image)
    return featureVector.data.numpy()[0]


def encodeBatch_deepfeat(data_batch):
    batch_encoded = []
    for row in data_batch['data']:
        img = data2image(row)
        batch_encoded.append(encodeImage_deepfeat(img))
    return np.array(batch_encoded)


def extractDeepFeat(dataset):
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=100, num_workers=30)

    model = models.alexnet(pretrained=True)
    newclassifier = torch.nn.Sequential(
        *list(model.classifier.children())[:-1])
    model.classifier = newclassifier

    trn_encoded = None
    labels = None
    for idx, (x, y) in enumerate(trainloader):
        print(idx + 1) * 100
        featureVector = model(x)
        if trn_encoded is None:
            trn_encoded = featureVector.data.numpy()
            labels = y.data.numpy()
        else:
            trn_encoded = np.concatenate(
                (trn_encoded, featureVector.data.numpy()))
            labels = np.concatenate((labels, y.data.numpy()))

    return trn_encoded, labels


if __name__ == '__main__':
    # transform_pipeline = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # # Train data
    # trainset = datasets.CIFAR10(
    #     root='.', train=True, download=False, transform=transform_pipeline)
    # trn_encoded, trn_labels = extractDeepFeat(trainset)
    # writePickle(trn_encoded, './deepfeat/trn_encoded')
    # writePickle(trn_labels, './deepfeat/trn_labels')

    # # Test data
    # testset = datasets.CIFAR10(
    #     root='.', train=False, download=False, transform=transform_pipeline)
    # tst_encoded, tst_labels = extractDeepFeat(testset)
    # writePickle(tst_encoded, './deepfeat/tst_encoded')
    # writePickle(tst_labels, './deepfeat/tst_labels')
    image = cv2.imread('./index.jpeg')
    encodeImage_deepfeat(image)
