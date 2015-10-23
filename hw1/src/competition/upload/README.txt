[Usage]
   gen.py   : run it in the fbank/ directory. It generates small training data/
   main.py  : starts the DNN training. It uses two other files
   |--train.py    : DNN structure and training
   |--readdata.py : read training data, test data, mappings, etc.

[Environment setting]
   Nothing special. We don't use GPU for training.

[Package dependency]
   Theano, Theano.tensor
