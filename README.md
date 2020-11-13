# Greeting_identifier
This package contains trained NLP models for identifying the presence of a greeting in a sentence. 
The two models are WE (a baseline model that make prediction from a gloVec word embeddings) and LSTM (an LSTM RNN). 
As expected the LSTM model outperforms the baseline model. Feel free to repurpose the code to solve other language
prediction problem. In fact, with a one-hot and a softmax activation implementation, this code that make predict
multiple classes from an input. Comments/suggestions are welcome.
