import pickle
from scipy.io import loadmat
from sklearn.utils import shuffle
import numpy as np



def y2indicator(y):
    N=len(y)
    ind=np.zeros((N,10))         #one hot encoding

    for i in range(N):
        ind[i,y[i]]=1
    return ind

#to rearrange the order of dimension (data is imported from matlab file)
#It makes the images grayscale
def rearrange(X):
    N=X.shape[-1]
    X_gray=X.mean(axis=2)
    out=np.zeros((N,32,32,1),dtype=np.float32)

    for i in range(N):
            out[i,:,:,0]=X_gray[:,:,i]
    return out


def normalize_data(Xtrain,Xtest,Xval):

    #Calculate mean and std
    train_mean = np.mean(Xtrain, axis=0)
    train_std = np.std(Xtrain, axis=0)

    #normalize the data
    Xtrain = (Xtrain - train_mean) / train_std
    Xtest = (Xtest - train_mean) / train_std
    Xval = (train_mean - Xval) / train_std

    return Xtrain,Xtest,Xval

def pickle_data():

    #load data
    train = loadmat('../large_training_data/train_32x32.mat')
    test = loadmat('../large_training_data/test_32x32.mat')

    #extract data
    Xtrain = rearrange(train['X'])
    Xtest = rearrange(test['X'])

    Ytrain = train['y'].flatten() % 10 # index in matlab starts from 1 (10 should be mapped on 0)
    Ytest = test['y'].flatten() % 10  # index in matlab starts from 1  (10 should be mapped on 0)



    del train, test

    #shuffle training data
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)


    #separate validation data
    Xval=Xtrain[70001:,:,:,:]
    Yval=Ytrain[70001:]

    Xtrain=Xtrain[:70000,:,:,:]
    Ytrain=Ytrain[:70000]

    #normalize data
    Xtrain, Xtest, Xval=normalize_data(Xtrain,Xtest,Xval)

    # encoding the labels
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    Yval_ind = y2indicator(Yval)



    #pickle data
    pickle.dump((Xtrain,Ytrain,Ytrain_ind), open("../../../large_training_data/train_32x32.p", "wb"))
    pickle.dump((Xtest,Ytest,Ytest_ind), open("../../../large_training_data/test_32x32.p", "wb"))
    pickle.dump((Xval, Yval, Yval_ind), open("../../../large_training_data/validation_32x32.p", "wb"))

if __name__=="__main__":
    pickle_data()