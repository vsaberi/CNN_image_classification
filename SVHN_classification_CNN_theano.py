import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from  theano.tensor.nnet import conv2d
from  theano.tensor.signal import pool

from scipy.io import loadmat
from sklearn.utils import shuffle

from datetime import datetime






def y2indicator(y):
    N=len(y)
    ind=np.zeros((N,10))         #one hot encoding

    for i in range(N):
        ind[i,y[i]]=1
    return ind

def error_rate(p,t):
    return np.mean(p != t)


def relu(a):
    return a*(a>0)


def convpool(X,W,b,poolsize=(2,2)):
    conv_out=conv2d(input=X,filters=W)

    pooled_out=pool.pool_2d(input=conv_out,ws=poolsize,ignore_border=True)

    return T.tanh(pooled_out+b.dimshuffle('x',0,'x','x'))


def init_filter(shape,poolsize):
    w=np.random.randn(*shape)/np.sqrt(np.prod(shape[1:]))+shape[0]*np.prod(shape[2:])/np.prod(poolsize)
    return w.astype(np.float32)

#to rearrange the order of dimension (data is imported from matlab file)
def rearrange(X):
    N=X.shape[-1]
    out=np.zeros((N,3,32,32),dtype=np.float32)

    for i in range(N):
        for j in range(3):
            out[i,j,:,:]=X[:,:,j,i]
    return out/255


def main():
    train = loadmat('../../../large_training_data/train_32x32.mat')
    test = loadmat('../../../large_training_data/test_32x32.mat')

    Xtrain = rearrange(train['X'])
    Ytrain = train['y'].flatten() - 1  # index in matlab starts from 1

    del train
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

    Ytrain_ind = y2indicator(Ytrain)

    Xtest = rearrange(test['X'])
    Ytest = test['y'].flatten() - 1  # index in matlab starts from 1
    del test
    Ytest_ind = y2indicator(Ytest)

    max_iter = 20
    print_period = 10

    lr=np.float32(0.0001)
    reg=np.float32(0.01)
    mu=np.float32(0.99)

    N=Xtrain.shape[0]

    batch_size = 500
    n_batches = int(N / batch_size)

    M = 500
    K = 10

    poolsize=(2,2)

    W1_shape=(20,3,5,5)
    W1_init = init_filter(W1_shape,poolsize)
    b1_init = np.zeros(W1_shape[0],dtype=np.float32)

    W2_shape = (50, 20, 5, 5)
    W2_init = init_filter(W2_shape, poolsize)
    b2_init = np.zeros(W2_shape[0], dtype=np.float32)


    W3_init = np.random.random((W2_shape[0]*5*5, M)) / np.sqrt(W2_shape[0]*5*5 + M)
    b3_init = np.zeros(M,dtype=np.float32)
    W4_init = np.random.random((M, K)) / np.sqrt(M + K)
    b4_init = np.zeros(K,dtype=np.float32)

    X=T.tensor4('X',dtype='float32')
    Y=T.matrix('T')

    W1=theano.shared(W1_init,'W1')
    b1=theano.shared(b1_init,'b1')
    W2=theano.shared(W2_init,'W2')
    b2=theano.shared(b2_init,'b2')
    W3=theano.shared(W3_init.astype(np.float32),'W3')
    b3=theano.shared(b3_init,'b3')
    W4=theano.shared(W4_init.astype(np.float32),'W4')
    b4=theano.shared(b4_init,'b4')


# to use momentum

    dW1=theano.shared(np.zeros(W1_init.shape,dtype=np.float32),'dW1')
    db1=theano.shared(np.zeros(b1_init.shape,dtype=np.float32),'db1')
    dW2=theano.shared(np.zeros(W2_init.shape,dtype=np.float32),'dW2')
    db2=theano.shared(np.zeros(b2_init.shape,dtype=np.float32),'db2')
    dW3=theano.shared(np.zeros(W3_init.shape,dtype=np.float32),'dW3')
    db3=theano.shared(np.zeros(b3_init.shape,dtype=np.float32),'db3')
    dW4=theano.shared(np.zeros(W4_init.shape,dtype=np.float32),'dW4')
    db4=theano.shared(np.zeros(b4_init.shape,dtype=np.float32),'db4')


    Z1=convpool(X,W1,b1)
    Z2=convpool(Z1,W2,b2)
    Z3=relu(Z2.flatten(ndim=2).dot(W3)+b3)
    pY=T.nnet.softmax(Z3.dot(W4)+b4)

    params=(W1,b1,W2,b2,W3,b3,W4,b4)

    reg_cost=reg*np.sum((param*param).sum() for param in params)
    cost=-(Y*T.log(pY)).sum()+reg_cost
    prediction=T.argmax(pY,axis=1)



    updated_W1 = W1 + mu * dW1 - lr * T.grad(cost,W1)
    updated_b1 = b1 + mu * db1 - lr * T.grad(cost,b1)
    updated_W2 = W2 + mu * dW2 - lr * T.grad(cost, W2)
    updated_b2 = b2 + mu * db2 - lr * T.grad(cost, b2)
    updated_W3 = W3 + mu * dW3 - lr * T.grad(cost, W3)
    updated_b3 = b3 + mu * db3 - lr * T.grad(cost, b3)
    updated_W4 = W4 + mu * dW4 - lr * T.grad(cost, W4)
    updated_b4 = b4 + mu * db4 - lr * T.grad(cost, b4)

    updated_dW1 = mu * dW1 - lr * T.grad(cost, W1)
    updated_db1 = mu * db1 - lr * T.grad(cost, b1)
    updated_dW2 = mu * dW2 - lr * T.grad(cost, W2)
    updated_db2 = mu * db2 - lr * T.grad(cost, b2)
    updated_dW3 = mu * dW3 - lr * T.grad(cost, W3)
    updated_db3 = mu * db3 - lr * T.grad(cost, b3)
    updated_dW4 = mu * dW4 - lr * T.grad(cost, W4)
    updated_db4 = mu * db4 - lr * T.grad(cost, b4)


    train=theano.function(
        inputs=[X,Y],
        allow_input_downcast=True,
        updates=[
            (W1,updated_W1),
            (b1, updated_b1),
            (W2, updated_W2),
            (b2, updated_b2),
            (W3, updated_W3),
            (b3, updated_b3),
            (W4, updated_W4),
            (b4, updated_b4),
            (dW1, updated_dW1),
            (db1, updated_db1),
            (dW2, updated_dW2),
            (db2, updated_db2),
            (dW3, updated_dW3),
            (db3, updated_db3),
            (dW4, updated_dW4),
            (db4, updated_db4)
        ]
    )

    get_prediction=theano.function(
        inputs=[X,Y],
        outputs=[cost,prediction],
        allow_input_downcast=True
    )

    t0=datetime.now()
    LL=[]


    test_cost = []


    for i in range(max_iter):
        print('iteration', str(i))
        for j in range(n_batches):
            Xbatch = Xtrain[j * batch_size:(j * batch_size + batch_size), ]
            Ybatch = Ytrain_ind[j * batch_size:(j * batch_size + batch_size), ]

            train(Xbatch,Ybatch)

            if j % print_period == 0:
                test_cost,predict_val=get_prediction(Xtest,Ytest_ind)
                print 'cost=', test_cost
                print 'error_rate=', error_rate(predict_val, Ytest)

    return test_cost


if __name__=='__main__':
    main()