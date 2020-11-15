#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 22:06:57 2020

@author: john.onwuemeka
"""

import streamlit as st
import numpy as np
import csv
import h5py
import mpu
from keras.models import Model,load_model,model_from_json
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def sigmoid(x):
    """Compute sigmoid values for each sets of scores in x."""
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
    """clean input sentence to remove special characters and punctuations
    
    Arguments:
    X -- input data containing sentences
    
    Returns:
    X -- cleaned version of input data
    """
    
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
    X = np.asarray([x if len(x.strip()) > 0 else 'empty input' for x in X])
    return X

              
def label_to_type(label):
    """
    Converts a label into prediction string
    """
    if label >= 0.5:
        return 'contains a Greeting'
    else:
        return 'does not contain a Greeting'
        

def predict(X, Y, W, b, word_to_vec_map):
    """
    Given X (sentences) and Y (greeting indicator), predict if it contains a 
    greeting and compute the accuracy of your model over the given set.
    
    Arguments:
    X -- input data containing sentences
    Y -- labels
    
    Returns:
    pred -- numpy array with your predictions
    """
    m = X.shape[0]
    pred = np.zeros((m, 1))
    
    for j in range(m):  # Loop over training examples
        
        # Split jth test example into list of lower case words
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
    avg -- average vector encoding information about the sentence
    """
    
    #split the words in the sentence
    words = [i.lower() for i in sentence.split()]
    
    #set the embeddings of unknown words prior to initialization
    try:
        fd = word_to_vec_map[words[0]]
    except:
        fd = word_to_vec_map['unknown']
        
    # Initialize the average word vector
    avg = np.zeros(fd.shape) 

    # average the word vectors
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

@st.cache()
def model_we(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
    
    """
    Model to train word vector representations in numpy.
    
    Arguments:
    X -- input data, numpy array of sentences as strings
    Y -- labels, numpy array of integers between 0 and 1
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations
    
    Returns:
    pred -- vector of predictions
    W -- weight matrix of the softmax layer
    b -- bias of the softmax layer
    """
    
    # Define number of training examples
    m = Y.shape[0]                          # number of training examples
    n_y = 1                                 # number of classes  
    n_h = 50                                # dimensions of the GloVe vectors 
    
    # Initialize parameters
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    
    # Optimization loop
    for t in range(num_iterations): # Loop over the number of iterations
        cost = 0
        for i in range(m):          # Loop over the training examples
            
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)
    
            # Forward propagate the avg through the sigmoid layer
            z = np.dot(W,avg)+b
            a = sigmoid(z)
    
            # Compute cost
            cost += -1*(Y[i]*np.log(a)[0] -(1-Y[i])*np.log(1-a))
            
            # Compute gradients 
            dz = a - Y[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db = dz
    
            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db
        
        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str((cost/m)[0]))
            pred,_ = predict(X, Y, W, b, word_to_vec_map) 

    return pred, W, b


def pel(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    # Initialize the embedding matrix as a numpy array of zeros.
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        
        #set the embeddings of unknown words
        try:
            fd = word_to_vec_map[word]
        except:
            fd = word_to_vec_map['unknown']
        emb_matrix[idx, :] = fd

    # Define Keras embedding layer with the correct input and output sizes
    embedding_layer = Embedding(input_dim=vocab_len,output_dim=emb_dim,trainable = False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


def s_2_i(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()`. 
    
    Arguments:
    X -- array of sentences 
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence.  
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X
    """
    
    m = X.shape[0]   # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):     # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words.
        sentence_words = [k.lower() for k in X[i].split()]
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            if j< max_len:
                
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                try:
                    fd = word_to_index[w]
                except:
                    fd = word_to_index['unknown']
                X_indices[i, j] = fd
                j += 1
                
    return X_indices

# @st.cache()
def model_lstm(X_train,Y_train,maxLen, word_to_vec_map, word_to_index):
    """    
    build a greeting identifier model
    
    Arguments:
    input_shape -- shape of the input
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary 
    X_train --- training input data set
    Y_train --- training labels

    Returns:
    model -- a model instance in Keras
    """
    

    # Define sentence_indices as the input of the graph.
    input_shape = (maxLen,)
    sentence_indices = Input(shape=input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pel(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer
    embeddings =  embedding_layer(sentence_indices)
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    X = LSTM(units = 128, return_sequences=True)(embeddings)
    
    # Add dropout with a probability of 0.7
    X = Dropout(rate = 0.7 )(X)
    
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    X = LSTM(units = 128, return_sequences=False)(X)
    
    # Add dropout with a probability of 0.7
    X = Dropout(rate = 0.7 )(X)
    
    # Propagate X through a Dense layer with 2 units
    X = Dense(1)(X)
    
    # Add a softmax activation
    X = Activation(activation='sigmoid')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    #compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    #prepare model input
    X_train_indices = s_2_i(X_train, word_to_index, maxLen)
    Y_train_oh = np.asarray([[i] for i in Y_train])
        
    #train the model for 20 epochs with minibatching
    model.fit(X_train_indices, Y_train_oh, epochs = 20, batch_size = 32, shuffle=True)
        
    
    return model

def load_variables(fname):
    x = mpu.io.read(fname)
    return x


def load_input():
        
    #load data set
    fname = './tagged_selections_by_sentence.csv'
    X_,Y_ = read_csv(fname)
    
    #clean-up dataset
    X_ = cleanX(X_)

    
    #split data set into training and test sets    
    ll = int(np.ceil(len(X_)*0.8))
    X_train,Y_train = X_[0:ll],Y_[0:ll]
    X_test,Y_test = X_[ll:],Y_[ll:]
        
    
    #load word embeddings
    w1 = load_variables('./word_to_vec_map_1.pickle')
    w2 = load_variables('./word_to_vec_map_2.pickle')
    word_to_vec_map = w1.copy()
    word_to_vec_map.update(w2)
    word_to_index = load_variables('./word_to_index.pickle')
    # index_to_word = load_variables('./index_to_word.pickle')
    
    return X_train,Y_train,X_test,Y_test,word_to_vec_map,word_to_index
    
@st.cache(suppress_st_warning=True)
def build_we_model():
    
    X_train,Y_train,X_test,Y_test,word_to_vec_map,_ = load_input()
    
    #train your model
    with st.spinner("Building WE model"):
        pred, W, b = model_we(X_train, Y_train, word_to_vec_map)
        st.text("Done building WE model ")
    
    #evaluate model performance
    with st.spinner("Evaluating model performance"):
        pred_train,accur_train = predict(X_train, Y_train, W, b, word_to_vec_map)
        pred_test,accur_test = predict(X_test, Y_test, W, b, word_to_vec_map)
        st.text("Accuracy of training set is: ")
        st.write('%.1f' % accur_train)
        st.text("Accuracy of test set is: ")
        st.write('%.1f' % accur_test)
        st.text("Done evaluating model performance")
    
    return W,b

@st.cache(suppress_st_warning=True)
def build_lstm_model():
    
    #set maxlen as the length of the longest sentence
    #set to 10 because a greeting would most likely be
    #within the first few sentences
    maxLen = 20 #len(max(X_train, key=len).split())
    
    X_train,Y_train,X_test,Y_test,word_to_vec_map,word_to_index = load_input()
    
    with st.spinner("Building lstm model"):
        model = model_lstm(X_train,Y_train,maxLen, word_to_vec_map, word_to_index)
        st.text("Done building lstm model ")
    
    X_test_indices = s_2_i(X_test, word_to_index, maxLen)
    Y_test_oh = np.asarray([[i] for i in Y_test])
    
    with st.spinner("Evaluating model performance"):
        loss, acc = model.evaluate(X_test_indices, Y_test_oh)
        st.text("Accuracy of test set is: ")
        st.write(round(acc,2))
        st.text("Done evaluating model performance")
    
    return model

def WE(build_model):
    
    if (build_model.lower()=='yes'):

        #build a model
        W,b = build_we_model()
        
        #load word vector map
        _,_,_,_,word_to_vec_map,_ = load_input()
        
        #request user input
        user_data = st.text_input("Enter sentence here: ",key="we_built")	
        #clean up input
        user_data = cleanX(np.array([user_data]))
        
        if user_data:
            #make prediction
            pred,_ = predict(user_data, np.array([1]), W, b, word_to_vec_map)
            out = label_to_type(pred[0])
            st.write('Probability is ',str(pred[0]))
            st.write('Your sentence ', out.lower())
            st.write('For an even better performance checkout the LSTM model')
            user_data = []
    else:
        #request user input
        user_data=[]
        user_data = st.text_input("Enter sentence here: ",key="we_loaded")
        if(user_data):
            
            #load pretrained weights
            filename = './trained_weights_we.h5'
            ft = h5py.File(filename,'r')
            W = ft['W']
            b = ft['b']

            #load word vector map
            _,_,_,_,word_to_vec_map,_ = load_input()
            
            #clean up input
            user_data = cleanX(np.array([user_data]))
            
            #make prediction
            pred,_ = predict(user_data, np.array([1]), W, b, word_to_vec_map)
            out = label_to_type(pred[0])
            st.write('Probability is ',str(pred[0]))
            st.write('Your sentence ', out.lower())
            st.write('For an even better performance, checkout the LSTM model')
            user_data=[]

def LSTM_RNN(build_model):
    
    if (build_model.lower()=='yes'):
        
        #buidl model and make predictions
        model = build_lstm_model()
    
        maxLen = 20
    
        #load word vector map
        _,_,_,_,_,word_to_index = load_input()
        
        #request user input
        user_data=[]
        user_data = st.text_input("Enter sentence here: ",key="lstm_built")
        
        if user_data:
            X_indices = s_2_i(cleanX(np.array([user_data])), word_to_index, maxLen)
            pred = model.predict(X_indices)
            out = label_to_type(pred[0])
            st.write('Probability is ',str(pred[0]))
            st.write('Your sentence ', out.lower()) 
            user_data = []
    else:
        #request user input
        user_data=[]
        user_data = st.text_input("Enter sentence here: ",key="lstm_loaded")
        try:           
            if (user_data):
                
                # load model
                fname = './trained_models_lstm.keras'
                model = load_model(fname)
                
                # # load json and create model
                # jf = open(fname.replace('h5','json'), 'r')
                # model_json = jf.read()
                # jf.close()
                # model = model_from_json(model_json)
            
                # # load weights into new model
                # model.load_weights(fname)
                
                maxLen = 20
                
                #load word vector map
                _,_,_,_,_,word_to_index = load_input()
            
                # make prediction
                X_indices = s_2_i(cleanX([user_data]), word_to_index, maxLen)
                pred = model.predict(X_indices)
                out = label_to_type(pred[0])
                st.write('Probability is ',str(pred[0]))
                st.write('Your sentence ', out.lower()) 
                user_data=[]
        except:
            
            st.write("whoops! I couldn't load the trained model into streamlit.")
            st.write("Will attempt again, otherwise I will build a model")
            # user_data = []
            
            # #request user input
            #user_data = st.text_input("Enter sentence here: ",key="lstm_built")

            if 	user_data:
                
                model = build_lstm_model()
            
                maxLen = 20
                #load word vector map
                _,_,_,_,_,word_to_index = load_input()
                
                X_indices = s_2_i(cleanX(np.array([user_data])), word_to_index, maxLen)
                pred = model.predict(X_indices)
                out = label_to_type(pred[0])
                st.write('Probability is ',str(pred[0]))
                st.write('Your sentence ', out.lower())                 
                user_data = []

            pass
          

def main():

    st.title("Identify the presence of a Greeting")
    
    build_model = st.sidebar.radio("Choose 'No' to use pretrained model or 'Yes' to train a model",['No','Yes'])
    optionss = [
        "WE",
        "RNN with LSTM"]

    page = st.sidebar.radio("Choose the NLP model", optionss)   
    # st.write(page)
    if page == 'WE':
        WE(build_model)
    elif page == 'RNN with LSTM':
        LSTM_RNN(build_model)

    
if __name__ == "__main__":
    main()