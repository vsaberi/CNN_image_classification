import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import picklew

from scipy.io import loadmat
from sklearn.utils import shuffle

from datetime import datetime








def error_rate(p,t):
    return np.mean(p != t)


def convpool(X,W,b):
    conv_out=tf.nn.conv2d(X,W,[1,1,1,1],padding='SAME')
    con_out=tf.nn.bias_add(conv_out,b)
    pool_out=tf.nn.max_pool(conv_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    return tf.nn.relu(pool_out)


def init_filter(shape,poolsize):
    w=np.random.randn(*shape)/np.sqrt(np.prod(shape[:-1]))+shape[-1]*np.prod(shape[:-2])/np.prod(poolsize)
    return w.astype(np.float32)

#to rearrange the order of dimension (data is imported from matlab file)





def main():

    Xtrain, Ytrain, Ytrain_ind = pickle.load(open("../../../large_training_data/train_32x32.p", "rb"))
    Xtest, Ytest, Ytest_ind = pickle.load(open("../../../large_training_data/test_32x32.p", "rb"))
    Xval, Yval, Yval_ind = pickle.load(open("../../../large_training_data/validation_32x32.p", "rb"))

    max_iter = 20
    print_period = 500

    lr=np.float32(0.0001)
    reg=np.float32(0.01)
    mu=np.float32(0.99)

    N=Xtrain.shape[0]

    batch_size = 500
    n_batches = int(N / batch_size)

    Xtrain=Xtrain[:73000,]
    Ytrain=Ytrain[:73000]
    Xtest=Xtest[:26000,]
    Ytest=Ytest[:26000]
    Ytest_ind=Ytest_ind[:26000,]



    M = 100
    K = 10

    poolsize=(2,2)

    W1_shape=(5,5,1,16)
    W1_init = init_filter(W1_shape,poolsize)
    b1_init = np.zeros(W1_shape[-1],dtype=np.float32)

    W2_shape = (5, 5,16,32)
    W2_init = init_filter(W2_shape, poolsize)
    b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

    W3_shape = (5, 5, 32, 64)
    W3_init = init_filter(W3_shape, poolsize)
    b3_init = np.zeros(W3_shape[-1], dtype=np.float32)


    W3_init = np.random.random((W2_shape[-1]*8*8, M)) / np.sqrt(W2_shape[-1]*8*8 + M)
    b3_init = np.zeros(M,dtype=np.float32)
    W4_init = np.random.random((M, K)) / np.sqrt(M + K)
    b4_init = np.zeros(K,dtype=np.float32)


    X=tf.placeholder(tf.float32,shape=(batch_size,32,32,1),name='X')
    T=tf.placeholder(tf.float32,shape=(batch_size,K),name='T')


    W1=tf.Variable(W1_init.astype(np.float32))
    b1=tf.Variable(b1_init.astype(np.float32))
    W2=tf.Variable(W2_init.astype(np.float32))
    b2=tf.Variable(b2_init.astype(np.float32))
    W3=tf.Variable(W3_init.astype(np.float32))
    b3=tf.Variable(b3_init.astype(np.float32))
    W4 = tf.Variable(W4_init.astype(np.float32))
    b4 = tf.Variable(b4_init.astype(np.float32))


    Z1 = convpool(X, W1, b1)
    Z2 = convpool(Z1, W2, b2)
    Z2_shape=Z2.get_shape().as_list()
    Z2_r=tf.reshape(Z2,[Z2_shape[0],np.prod(Z2_shape[1:])])
    Z3 = tf.nn.relu(tf.matmul(Z2_r,W3)+ b3)
    Yish = tf.matmul(Z3,W4)+ b4

    cost=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish,labels=T))

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.005, global_step, 10000, 0.95)
    train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost, global_step=global_step)


    # train_op=tf.train.RMSPropOptimizer(0.0001,decay=0.0,momentum=0.9).minimize(cost)


    predict_op=tf.argmax(Yish,1)

    t0=datetime.now()

    LL=[]

    init=tf.initialize_all_variables()







    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            print('iteration', str(i))
            for j in range(n_batches):
                Xbatch = Xtrain[j * batch_size:(j * batch_size + batch_size), ]
                Ybatch = Ytrain_ind[j * batch_size:(j * batch_size + batch_size), ]

                if len(Xbatch)==batch_size:
                    session.run(train_op,feed_dict={X:Xbatch,T:Ybatch})
                    if j % print_period==0:

                        test_cost=0
                        prediction=np.zeros(len(Xtest))
                        for k in range(int(len(Xtest)/batch_size)):
                            Xtestbatch = Xtest[k * batch_size:(k * batch_size + batch_size), ]
                            Ytestbatch = Ytest_ind[k * batch_size:(k * batch_size + batch_size), ]
                            test_cost+=session.run(cost,feed_dict={X:Xtestbatch,T:Ytestbatch})
                            prediction[k * batch_size:(k * batch_size + batch_size)]=session.run(

                                predict_op,feed_dict={X:Xtestbatch})

                        err=error_rate(prediction,Ytest)

                        print("Cost/err at iteration i=%d, j=%d: %.3f/%.3f" % (i,j,test_cost,err))
                        LL.append(test_cost)

    print("Elapsed time:",(datetime.now()-t0))
    plt.plot(LL)
    plt.show()




if __name__ == '__main__':
    main()