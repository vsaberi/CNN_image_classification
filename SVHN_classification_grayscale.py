import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle


def flatten(X):
    N=X.shape[-1]           #number of images (# rows in matlab file)
    flat=np.zeros((N,32*32*1))
    # X_gray=np.zeros((N,32,32))
    # for i in range(N):
    #     for j in range(32):
    #         for k in range(32):
    #             X_gray[j,k,i]=np.mean(X[j,k,:,i])
    # for i in range(N):
    X_gray=X.mean(axis=2)
    for i in range(N):
        flat[i]=X_gray[:,:,i].reshape(32*32*1)

    return flat


def y2indicator(y):
    N=len(y)
    ind=np.zeros((N,10))         #one hot encoding

    for i in range(N):
        ind[i,y[i]]=1
    return ind

def error_rate(p,t):
    return np.mean(p != t)






def main():
    train=loadmat('../../../large_training_data/train_32x32.mat')
    test=loadmat('../../../large_training_data/test_32x32.mat')



    Xtrain=flatten(train['X'].astype(np.float32)/255)
    Ytrain=train['y'].flatten()-1       #index in matlab starts from 1
    Xtrain,Ytrain=shuffle(Xtrain,Ytrain)

    Ytrain_ind=y2indicator(Ytrain)

    Xtest = flatten(test['X'].astype(np.float32) / 255)
    Ytest = test['y'].flatten() - 1  # index in matlab starts from 1
    Ytest_ind = y2indicator(Ytest)

    max_iter=20
    print_period=500
    N,D=Xtrain.shape

    batch_size=500
    n_batches=int(N/batch_size)

    M1=1000
    M2=500
    K=10

    W1_init=np.random.random((D,M1))/np.sqrt(D+M1)
    b1_init=np.zeros(M1)
    W2_init = np.random.random((M1, M2)) / np.sqrt(M1 + M2)
    b2_init = np.zeros(M2)
    W3_init = np.random.random((M2,K)) / np.sqrt(M2 + K)
    b3_init = np.zeros(K)


    X=tf.placeholder(tf.float32,shape=(None,D),name='X')
    T=tf.placeholder(tf.float32,shape=(None,K),name='T')
    W1=tf.Variable(W1_init.astype(np.float32))
    b1=tf.Variable(b1_init.astype(np.float32))
    W2=tf.Variable(W2_init.astype(np.float32))
    b2=tf.Variable(b2_init.astype(np.float32))
    W3=tf.Variable(W3_init.astype(np.float32))
    b3=tf.Variable(b3_init.astype(np.float32))

    Z1=tf.nn.relu(tf.matmul(X,W1)+b1)
    Z2=tf.nn.relu(tf.matmul(Z1,W2)+b2)
    Yish=tf.matmul(Z2,W3)+b3

    cost=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish,labels=T))

    train_op=tf.train.RMSPropOptimizer(0.0001,decay=0.0,momentum=0.9).minimize(cost)

    predict_op=tf.argmax(Yish,1)

    LL=[]

    init=tf.initialize_all_variables()

    test_cost=[]
    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            print('iteration',str(i))
            for j in range(n_batches):
                Xbatch=Xtrain[j*batch_size:(j*batch_size+batch_size),]
                Ybatch=Ytrain_ind[j*batch_size:(j*batch_size+batch_size),]

                session.run(train_op,feed_dict={X:Xbatch,T:Ybatch})

                if j%print_period==0:
                    test_cost.append(session.run(cost,feed_dict={X:Xtest,T:Ytest_ind}))
                    prediction=session.run(predict_op,feed_dict={X:Xtest})
                    print('cost=',test_cost[-1])
                    print('error_rate=',error_rate(prediction,Ytest))





    return test_cost






if __name__=='__main__':
    test_c=main()

    plt.plot(test_c)
    plt.show()