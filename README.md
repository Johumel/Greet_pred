# Greeting_identifier
This package contains trained NLP models for identifying the presence of a greeting in a sentence. 
The two models are WE (a baseline model that make prediction from a gloVec word embeddings) and LSTM (a LSTM many-to-one RNN). 
As expected the LSTM model outperforms the baseline model. Feel free to repurpose the code to solve other language
prediction problem. In fact, with a one-hot and a softmax activation implementation, this code can make predict for
multiple classes. The dataset used for training the models was downloaded from 
https://nextit-public.s3-us-west-2.amazonaws.com/. The GloVec word embeddings was downloaded from 
https://github.com/uclnlp/inferbeddings/tree/master/data/glove. Comments/suggestions are welcome.
