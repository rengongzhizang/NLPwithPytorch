#  NLPwithPyTorch
## Introduction
A Set of  Latinic Language Binary Classifiers including simple Bag of Words, Learned Word Embeddings, GloVe Word Embeddings, vanilla RNN, and LSTM (and also a great tutorial to learn PyTorch and review basic NLP concepts!)

## Structures
Their structures are shown as follows:

Bag of Words: Bag of Words -> Linear -> sigmoid -> logistic loss

Learned Word Embeddings: Word Embeddings -> Average Pooling -> Linear -> sigmoid -> logistic loss

Pretrained Word Embeddings: GloVe Word Embedding -> Average Pooling -> Linear -> sigmoid -> logistic loss

vanilla RNN: GloVe Word Embedding -> Average Pooling ->  RNN -> Linear -> sigmoid -> logistic loss

LSTM: GloVe Word Embedding -> Average Pooling ->  LSTM -> Linear -> sigmoid -> logistic loss

## How to run:
open a teminal and:
```shell
python3 textClassification.py
```
