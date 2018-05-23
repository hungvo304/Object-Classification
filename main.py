from utility import loadPickle

if __name__ == '__main__':
    databatch1 = loadPickle('./cifar-10-batches-py/data_batch_1')
    print databatch1['data'].shape
    print len(databatch1['labels'])
    label_names = loadPickle('./cifar-10-batches-py/batches.meta')
    print label_names
