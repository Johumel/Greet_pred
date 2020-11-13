#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 22:06:57 2020

@author: john.onwuemeka
"""

import streamlit as st
import numpy as np
import csv
import mpu

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        i = 0
        for line in f:
            if i>1:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                if curr_word.lower() == 'unknown':
                    print(i)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            i+=1
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read_csv(fname):
    text = []
    classy = []

    with open (fname) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            text.append(row[6])
            classy.append(row[7])

    X = np.asarray(text)
    Y = np.asarray(classy, dtype=int)

    return X, Y


def cleanX(X):
    for h in range(len(X)):
        i = X[h]
        nh = i.split()
        new_nh = []
        for j in nh:
            j = j.strip("..,)(:?$#@&!;")
            if "'" in j:
                j = j.replace("'","")
            if '.' in j:
                j = j.split('.')
            elif '/' in j:
                j = j.split('/')
            elif '-' in j:
                j = j.split('-')
            elif j == 'speedbird':
                j = [j[0:5],j[5:]]
            if isinstance(j,list):
                new_nh.append(j[0])
                new_nh.append(j[1])
            else:
                new_nh.append(j)
        X[h] = ' '.join(new_nh)
    return X

              
def label_to_type(label):
    if label >= 0.5:
        return 'contains a Greeting'
    else:
        return 'does not contain a Greeting'
    
def print_predictions(X, pred):
    print()
    for i in range(X.shape[0]):
        print(X[i], label_to_type(float(pred[i])))
        

def predict(X, Y, W, b, word_to_vec_map):
    """
    Given X (sentences) and Y (emoji indices), predict emojis and compute the accuracy of your model over the given set.
    
    Arguments:
    X -- input data containing sentences, numpy array of shape (m, None)
    Y -- labels, containing index of the label emoji, numpy array of shape (m, 1)
    
    Returns:
    pred -- numpy array of shape (m, 1) with your predictions
    """
    m = X.shape[0]
    pred = np.zeros((m, 1))
    
    for j in range(m):                       # Loop over training examples
        
        # Split jth test example (sentence) into list of lower case words
        words = X[j].lower().split()
        
        # Average words' vectors
        avg = np.zeros((50,))
        for w in words:
            try:
                fd = word_to_vec_map[w]
            except:
                fd = word_to_vec_map['unknown']
            avg += fd
        avg = avg/len(words)

        # Forward propagation
        Z = np.dot(W, avg) + b
        A = sigmoid(Z)
        pred[j] = A
    predictions = np.asarray([1 if i[0] >=0.5 else 0 for i in pred])
    accur = np.mean(predictions[:] == Y[:])
    
    return pred,accur

def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.
    
    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """
    
    #split the words in the sentence
    words = [i.lower() for i in sentence.split()]
    
    #set the embeddings of unknown words prior to initialization
    try:
        fd = word_to_vec_map[words[0]]
    except:
        fd = word_to_vec_map['unknown']
        
    # Initialize the average word vector, should have the same shape as your word vectors.
    avg = np.zeros(fd.shape) 

    # average the word vectors..
    total = np.zeros(avg.shape).tolist()
    for w in words:
        #set the embeddings of unknown words
        try:
            fd = word_to_vec_map[w]
        except:
            fd = word_to_vec_map['unknown']
        total += fd

    avg = np.asarray(total/len(words))
    
    return avg

#Build the predictor model
def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 300):
    """
    Model to train word vector representations in numpy.
    
    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations
    
    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """
    
    np.random.seed(1)

    # Define number of training examples
    m = Y.shape[0]                          # number of training examples
    n_y = 1                                 # number of classes  
    n_h = 50                                # dimensions of the GloVe vectors 
    
    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    # Convert Y to Y_onehot with n_y classes
    Y_oh = Y
    
    # Optimization loop
    for t in range(num_iterations): # Loop over the number of iterations
        cost = 0
        for i in range(m):          # Loop over the training examples
            
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer
            z = np.dot(W,avg)+b
            a = sigmoid(z)

            # Compute cost
            cost += -1*(Y_oh[i]*np.log(a)[0] -(1-Y_oh[i])*np.log(1-a))
            
            # Compute gradients 
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db
        
        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str((cost/m)[0]))
            pred,_ = predict(X, Y, W, b, word_to_vec_map) #predict is defined in emo_utils.py

    return pred, W, b


def load_variables(fname):
    x = mpu.io.read(fname)
    return x

def main():
    st.title("Identify the presence of Greetings")
    
    #load data set
    #dataset downloaded from https://nextit-public.s3-us-west-2.amazonaws.com/
    fname = './tagged_selections_by_sentence.csv'
    
    X_,Y_ = read_csv(fname)
    
    #clean-up dataset
    X_ = cleanX(X_)
    X_ = np.asarray([x if len(x.strip()) > 0 else 'Just dont know' for x in X_ ])
    
    #split data set into training and test sets    
    ll = int(np.ceil(len(X_)*0.8))
    X_train,Y_train = X_[0:ll],Y_[0:ll]
    X_test,Y_test = X_[ll:],Y_[ll:]
        
    #load pre-trained word embeddings
    #word embeddings download from https://github.com/uclnlp/inferbeddings/blob/master/data/glove/
    #fname2 = './glove.6B.50d.dat'
    
    # word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(fname2)
    
    w1 = load_variables('./word_to_vec_map_1.pickle')
    w2 = load_variables('./word_to_vec_map_2.pickle')
    word_to_vec_map = w1.update(w2)
    st.write(word_to_vec_map['unknown'])
    
    build_model = st.sidebar.checkbox("Check to build model otherwise pretrained model will be loaded")
    
    if(build_model):
            
        #train your model
        st.text("Building model .... ")
        pred, W, b = model(X_train, Y_train, word_to_vec_map)
        st.text("Done building model ")
        
        #evaluate model performance
        st.text("Evaluating model performance .... ")
        pred_train,accur_train = predict(X_train, Y_train, W, b, word_to_vec_map)
        pred_test,accur_test = predict(X_test, Y_test, W, b, word_to_vec_map)
        st.text("Accuracy of training set is: ")
        st.write('%.1f' % accur_train)
        st.text("Accuracy of test set is: ")
        st.write('%.1f' % accur_test)
        st.text("Done evaluating model performance")

        #request user input
        user_data = st.text_input("Enter sentence here: ")	
        
        #clean up input
        user_data = cleanX(np.array([user_data]))
        
        #make prediction
        pred,_ = predict(user_data, np.array([1]), W, b, word_to_vec_map)
        out = label_to_type(pred[0])
        st.write('Your sentence ', out.lower())
        build_model = False
    
    else:
        
        #load pretrained parameters
        filename = './trained_model_params.pickle'
    
        gh = mpu.io.read(filename)
        W = gh['W']
        b = gh['b']
        
        #request user input
        user_data=[]
        user_data = st.text_input("Enter sentence here: ")
        if(user_data):
            
            #clean up input
            user_data = cleanX(np.array([user_data]))

            #make prediction
            pred,_ = predict(user_data, np.array([1]), W, b, word_to_vec_map)
            out = label_to_type(pred[0])
            st.write('Your sentence ', out.lower())
        
if __name__ == "__main__":
    main()
