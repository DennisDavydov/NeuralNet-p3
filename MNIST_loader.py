import _pickle
import pickle
import gzip
import numpy as np
import os
import imutils
import cv2


def load_data():
    filepath =os.path.join(os.path.dirname(__file__), 'mnist.pkl.gz')
    f = gzip.open(filepath, 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)
    
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    
    TR_D = []
    TR_R = []
    for im, res in zip(tr_d[0], tr_d[1] ):
        im = np.reshape(im, (784,1))
        im = np.reshape(im, (28,28))
        #cv2.imshow('im', np.reshape(im, (28,28))*255)
        #cv2.waitKey(50)
        for angle in range(-30, 30, 30):
            TR_D.append(imutils.rotate(im, angle))
            TR_R.append(res)
    tr_d = TR_D
    tr_r = TR_R
            
    
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d]
    training_results = [vectorized_result(y) for y in tr_r]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    print('data loaded...')
    return (training_data, validation_data, test_data)
    
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e