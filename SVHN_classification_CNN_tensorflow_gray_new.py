import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import time
from sklearn.utils import shuffle
import pickle
from datetime import datetime
import os








def plot_images(X_images, num_rows, num_cols, Y_ind_true, Y_ind_pre=None):
    """ This function plots a grid of num_rows*num_cols images with their true and predicted labels
    """
    # Initialize a num_rows*num_cols grid
    _, axes = plt.subplots(num_rows, num_cols)

    # Randomly select num_rows * num_cols images
    rs = np.random.choice(X_images.shape[0], num_rows * num_cols)

    # For every axes object in the grid
    for image_index, ax in zip(rs, axes.flat):

        # Predictions=Null
        if Y_ind_pre is None:
            title = "True: {}".format(np.argmax(Y_ind_true[image_index]))

        # Prediction is given
        else:
            title = "True: {}, Pred: {}".format(np.argmax(Y_ind_true[image_index]), Y_ind_pre[image_index])

        # Display the image
        ax.imshow(X_images[image_index, :, :, 0], cmap='gray')

        ax.set_title(title)

        # Do not overlay a grid
        ax.set_xticks([])
        ax.set_yticks([])


def W_variable(name,shape,type="fc"):
    """"Returns convolutional layer weight. If the name already exists it just retrieves it
    type= "conv" or "fc"
    """
    if type=="conv":
        w=tf.get_variable(name=name,shape=shape,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32))
    elif type=="fc":
        w= tf.get_variable(name=name,shape=shape,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    return w


def b_variable(shape):
    """"Returns a bias variable
    """
    return tf.Variable(tf.constant(0.0,shape=shape))


def conv_layer(input,                   #input to the layer
               layer_name,                    #layer name
               num_input_channels,      #number of channels in previous layer
               num_filters,             #number of filters in the layer
               filter_size=5,             #filter dimension (W*H)
               pooling_size=2,           #max pooling layer size (2*2). "None" removes pooling layer
               dropout_keep_prob=None    #dropout keep probablity (None: no dropout)
               ):

    #filter weight shape
    shape=[filter_size,filter_size,num_input_channels,num_filters]

    #create weight
    W=W_variable(name=layer_name,shape=shape,type="conv")

    #create bias
    b=b_variable(shape=[num_filters])

    #output operation
    output=tf.nn.conv2d(input=input,
                         filter=W,
                         strides=[1,1,1,1],
                         padding="SAME")

    #apply bias
    output+=b

    #activation
    output = tf.nn.relu(output)


    #pooling (down-sampling)
    if pooling_size is not None:
        output=tf.nn.max_pool(value=output,
                              ksize=[1,pooling_size,pooling_size,1],
                              strides=[1,pooling_size,pooling_size,1],
                              padding="SAME")
    if dropout_keep_prob is not None:
        output=tf.nn.dropout(output,keep_prob=dropout_keep_prob)

    return output,W



def flatten(input):
    """This function flattens the convolutional leyrs output to feed to fully-connected layer
    """

    #find the shape
    shape=input.get_shape()

    num_features=np.prod(shape[1:])

    #reshape
    output=tf.reshape(input, [-1, num_features])

    return output,num_features


def fc_layer(input,                   #input to the fully-connected layer
               layer_name,                    #layer name
               num_inputs,      #number of channels in previous layer
               num_outputs,             #number of filters in the layer
               activation="relu",       #options: "sigmoid", "relu", "None"
               dropout_keep_prob = None  # dropout keep probablity (None: no dropout)
            ):

    #create weight
    W=W_variable(name=layer_name,shape=[num_inputs,num_outputs],type="fc")

    #create bias
    b=b_variable(shape=[num_outputs])

    #output operation
    output=tf.matmul(input,W)

    output+=b


    #pooling (down-sampling)
    if activation=="relu":
        output=tf.nn.relu(output)
    elif activation=="sigmoid":
        output=tf.nn.sigmoid(output)


    if dropout_keep_prob is not None:
        output=tf.nn.dropout(output,keep_prob=dropout_keep_prob)

    return output,W



def model(X_input,  #input to model
          keep_prob,  # dropout keep probablity (after second conv)
          filt_sz1=5,  #filter size layer 1 defult:(5*5)
          num_filt1=32,  #number of filters layer 1
          filt_sz2=5,  #filter size layer 2 defult:(5*5)
          num_filt2=32,  #number of filters layer 2
          fc_sz=256,  #number of neurons of dense layer
          num_classes=10  #number of classes
          ):

    output_con_1, w_c1 = conv_layer(X_input,
                              layer_name="conv_1",
                              num_input_channels=1,
                              num_filters=num_filt1,
                              filter_size=filt_sz1)

    output_con_2, w_c2 = conv_layer(output_con_1,
                              layer_name="conv_2",
                              num_input_channels=num_filt1,
                              num_filters=num_filt2,
                              filter_size=filt_sz2,
                              dropout_keep_prob=keep_prob)

    output_flatten,num_features = flatten(output_con_2)


    output_fc_1,_=fc_layer(output_flatten,
                    layer_name="fc_1",
                    num_inputs=num_features,
                    num_outputs=fc_sz,
                    activation="relu")

    output_fc_2,_ = fc_layer(output_fc_1,
                      layer_name="fc_2",
                      num_inputs=fc_sz,
                      num_outputs=num_classes,
                      activation=None)

    return output_fc_2


def train(X_data,                   #X data(training images)
          Y_ind_data,               #encoded labels (indicators)
          X_val,                    #validation data
          Y_ind_val,                #Y_ind validation
          keep_prob,                #dropout keep prob
          num_iterations,           #number of iterations
          batch_sz,                 #batch size (for each gradient descent operation)
          print_period,             #print period
          train_op,                 #training optimization operation
          accuracy_calc,            #accuracy calculator
          tf_session,                #tensorflow session
          x,
          y,
          keep,
          global_step
          ):
    start_time=time.time()

    for i in range(num_iterations):

        #set batch data
        offset=(i*batch_sz)%(X_data.shape[0]-batch_sz)
        X_batch=X_data[offset:(offset+batch_sz),:,:,:]
        Y_ind_batch=Y_ind_data[offset:(offset+batch_sz),:]

        #train
        tf_session.run(train_op,feed_dict={x:X_batch,y:Y_ind_batch,keep:keep_prob})

        if i % print_period==0:
            batch_acc=tf_session.run(accuracy_calc,feed_dict={x:X_batch,y:Y_ind_batch,keep:keep_prob})
            print('batch accuracy at step %d:%.4f'%(i,batch_acc))

            val_acc=tf_session.run(accuracy_calc,feed_dict={x:X_val,y:Y_ind_val,keep:1.0})
            print('validation accuracy at step %d:%.4f'%(i,val_acc))

            #save session
            tf.train.Saver().save(tf_session, 'saved_sessions/', global_step=global_step)


def main():

    Xtrain, Ytrain,Ytrain_ind = pickle.load(open("../large_training_data/train_32x32.p", "rb"))
    Xtest, Ytest,Ytest_ind = pickle.load(open("../large_training_data/test_32x32.p", "rb"))
    Xval, Yval,Yval_ind = pickle.load(open("../large_training_data/validation_32x32.p", "rb"))

    # print("Training set",Xtrain.shape,Ytrain.shape)
    # print("Testing set",Xtest.shape,Ytest.shape)
    # print("Validation set",Xval.shape,Yval.shape)



    #define placeholders
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name='x')
    y = tf.placeholder(tf.float32, shape=(None, 10), name='y')
    y_ind=tf.cast(tf.argmax(y,dimension=1),dtype=tf.float32)
    keep=tf.placeholder(tf.float32,name='keep')


    #model output
    logits=model(X_input=x,keep_prob=keep)



    # training optimization operation with decaying learning rate
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.96, staircase=True)
    train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost, global_step=global_step)

    # accuracy calculation operation
    Y_ind_prediction = tf.cast(tf.argmax(logits, dimension=1),tf.float32)

    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(Y_ind_prediction, y_ind), tf.float32))



    #define session and initialize
    session = tf.Session()
    session.run(tf.initialize_all_variables())

    train(X_data=Xtrain,
          Y_ind_data=Ytrain_ind,
          X_val=Xval,
          Y_ind_val=Yval_ind,
          keep_prob=0.5,
          num_iterations=5000,
          batch_sz=100,
          print_period=500,
          train_op=train_op,
          accuracy_calc=accuracy_op,
          tf_session=session,
          x=x,
          y=y,
          keep=keep,
          global_step=global_step)
    # Plot 2 rows with 9 images each from the training set
    plot_images(Xtrain, 2, 9, Ytrain_ind)
    plt.show()





if __name__ == '__main__':
    main()
