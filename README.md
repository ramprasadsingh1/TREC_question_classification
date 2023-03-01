# TREC Question Classification

The main source code is in question_classifier.py

## Running the code
To run the code download the folder, which has the 

## Changing the settings
The settings for model training and testing can be set by editing config.yml
This file contains the following fields that will only accept the below mentioned values

path_train: data/train.label 
path_test: data/test.label
classification: coarse/fine (choose between Coarse or Fine classification)   
model: bow/bilstm (choose between BoW or BiLSTM model)
embedding: random/glove (choose between Random or Glove embedding)
setting: freeze/finetune (choose between Freeze and Fine Tuning the settings)


To train a model use the following command:
& python3 question_classifier.py --train --config [configuration_file_path]
We should also be able to run your code to test a model by issuing the following command:
% python3 question_classifier.py --test --config [configuration_file_path]
The program will load a configuration file storing all needed information in different sections, such as:
# Paths To Datasets And Evaluation
path_train : ../data/train.txt
2
path_dev : ../data/dev.txt
path_test : ../data/test.txt
classes: coarse
# Options for model
model : bow # bow, bilstm, bow_ensemble, bilstm_ensemble...
path_model : ../data/model.bow
# Early Stopping
early_stopping : 5
# Model Settings
epoch : 10
lowercase : false
# Using pre-trained Embeddings
path_pre_emb : ../data/glove.txt
# Network Structure
word_embedding_dim : 200
batch_size : 20
# Hyperparameters
lr_param : 0.0001
# Evaluation
path_eval_result : ../data/output.txt
# Ensemble model
model : bilstm_ensemble
ensemble_size : 5
path_model : ../data/model.bilstm_ensemble
Notes:
- If your code supports more than two required models (as mentioned in Section 2.5), such as an
ensemble of 5 BiLSTM models, your configuration file may include the ‘Ensemble model’ section
as in the above example. In that case, the five BiLSTM models will be stored in
../data/model.bilstm_ensemble.0
../data/model.bilstm_ensemble.1
...
../data/model.bilstm_ensemble.4
- Output (e.g., ../data/output.txt) is a file in which each line is a class for each testing question
and the performance (i.e., accuracy).
- You may need to store some more information on the model (e.g., vocabulary). Do not hesitate to
make use of the configuration file to specify paths to any information you might wish to store.
