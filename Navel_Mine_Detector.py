#Naval Mine Detector Application
'''
In this use case, they have been provided with a SONAR data set which contains the
data about 208 patterns obtained by bouncing sonar signals off a metal cylinder (naval
mine) and a rock at various angles and under various conditions.
Now, as we know, a naval mine is a self-contained explosive device placed in water to
damage or destroy surface ships or submarines.
So, our goal is to build a model that can predict whether the object is a naval mine or
rock based on our data set.
'''

'''
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Description: Naval Mine Detector Application

Author: Aishwarya 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

############################################################################

#Reading the dataset
def read_dataset():
    df = pd.read_csv("sonar.csv")
    print("Dataset loaded succesfully")
    print("Number of columns:",len(df.columns))

    #features of dataset
    X = df[df.columns[0:60]].values

    #label of dataset
    y = df[df.columns[60]]

    #encode the dependant variable
    encoder = LabelEncoder()

    #encode character labeles into integer i.e 1 or 0 (one hot encode)
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)

    print("X.shape",X.shape)

    return (X,Y)

#################################################################################

#define the encoder function to set M => 1, R=> 0
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels] = 1
    return one_hot_encode

###############################################################################

#Model for training
def multilayer_perceptron(x, weights, biases):
    #hidden layer with RELU activations
    #first layer performs matrix multiplication of input with weights
    layer_1 = tf.add(tf.matmul(x,weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    #hidden layer with sigmod activations
    #second layer performs matrix multiplication of layer1 with weights
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    #hidden layer with sigmoid activations
    #third layer performs matrix multiplication of layer2 with weights
    layer_3 = tf.add(tf.matmul(layer_2,weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    #hidden layer with RELU activations
    #Forth layer performs matrix multiplication of layer3 with weights
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    #output layer with linear activations
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer

#######################################################################

def main():
    #read the dataset
    X, Y = read_dataset()

    #suffle the dataset to mix up the rows
    X, Y = shuffle(X, Y, random_state = 1)

    #covert the dataset into train and test datasets
    #for testing we use 10% and for traning we use 90%
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.30, random_state=415)

    #inspect the shape of the train and test datasets
    print("train_x.shape", train_x.shape)
    print("train_y.shape", train_y.shape)
    print("test_x.shape", test_x.shape)
    print("test_y.shape", test_y.shape)

    #define the para which are reqd for rensors i.e hyperparaemets
    #change in a variable in each iteration
    learning_rate = 0.3

    #total number of iterations to minimize the error
    training_epochs = 1000

    cost_history = np.empty(shape=[1], dtype=float)

    #no of features <=> no of columns
    n_dim = X.shape[1]
    print("No of col are n_dim", n_dim)

    #as we have two classes as r and m
    n_class = 2

    #path which contains model files
    model_path = "Aishwarya"

    n_hidden_1 = 60
    n_hidden_2 = 60
    n_hidden_3 = 60
    n_hidden_4 = 60

    tf.compat.v1.disable_eager_execution()
    
    x = tf.compat.v1.placeholder(tf.float32,[None, n_dim])
    y_ = tf.compat.v1.placeholder(tf.float32,[None, n_class])

    W = tf.Variable(tf.zeros([n_dim, n_class]))
    b = tf.Variable(tf.zeros([n_class]))

    weights = {
        'h1': tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_1])),
        'h2': tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_2])),
        'h3': tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_3])),
        'h4': tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_4])),
        'out': tf.Variable(tf.random.truncated_normal([n_hidden_4,n_class])),
    }

    biases = {
        'b1': tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_1])),
        'b2': tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_2])),
        'b3': tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_3])),
        'b4': tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_4])),
        'out': tf.Variable(tf.random.truncated_normal([n_class])),
    }

    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    y = multilayer_perceptron(x,weights,biases)

    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    traning_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    sess = tf.compat.v1.Session()
    sess.run(init)

    # calculate the cost and accuracy for each epoch

    mse_history = []
    accuracy_history = []

    for epoch in range(training_epochs):
        sess.run(traning_step, feed_dict={x: train_x, y_: train_y})
        cost = sess.run(cost_function, feed_dict={x:train_x, y_:train_y})
        cost_history = np.append(cost_history, cost)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pred_y = sess.run(y,feed_dict={x:test_x})
        mse = tf.reduce_mean(tf.square(pred_y - test_y))
        mse_ = sess.run(mse)
        accuracy = (sess.run(accuracy, feed_dict={x:train_x, y_:train_y}))
        accuracy_history.append(accuracy)
        print('epoch:',epoch,'-','cost:',cost,"- MSE:",mse_,"- Train Accuracy:",accuracy)


    #model gets saved in the file
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s", save_path)

    #display graph for accracy history
    plt.plot(accuracy_history)
    plt.title("Accuracy History")
    plt.xlab('Eppoch')
    plt.ylabel('Loss')
    plt.show()

    #print the final mean square error
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.square(pred_y - test_y))
    print("Test accuracy:", (sess.run(y, feed_dict={x:test_x, y_:test_y})))

    #print the final mean square error
    pred_y = sess.run(y,feed_dict={x:test_x})
    mse = tf.reduce_mean(tf.square(pred_y- test_y))
    print("Mean square errpr: %.4f" % sess.run(mse))

if __name__ == "__main__":
    main()



















