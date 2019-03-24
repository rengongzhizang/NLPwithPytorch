import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from textClassifiers import *
from textClassificationDatasets import *
import time
import copy
from utils import *

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ''' loading data '''
    train_sen, train_lab = fileLoader('data/train.txt')
    val_sen, val_lab = fileLoader('data/dev.txt')
    test_sen, test_lab = fileLoader('data/test.txt')
    unlab_sen = fileLoader('data/unlabelled.txt', with_label=False)
    
    ''' building the bag of words dictionary'''
    word_to_idx = make_words_dict(train_sen, val_sen, test_sen, unlab_sen)

    ''' generating vector for bag of words '''
    #train_vec = make_bow_vector(train_sen, word_to_idx)
    #val_vec = make_bow_vector(val_sen, word_to_idx)
    #test_vec = make_bow_vector(test_sen, word_to_idx)
    #unlab_vec = make_bow_vector(unlab_sen, word_to_idx)

    ''' generating vector for words embedding'''
    emb_train_vec = make_embed_vector(train_sen, word_to_idx)
    emb_val_vec = make_embed_vector(val_sen, word_to_idx)
    emb_test_vec = make_embed_vector(test_sen, word_to_idx)
    emb_unlab_vec = make_embed_vector(unlab_sen, word_to_idx)
    
    #bagOfWordClassification(device, train_vec, val_vec, test_vec, unlab_vec, train_lab, val_lab, test_lab)
    #wordEmbeddingClassificationLearned(device, emb_train_vec, emb_val_vec, emb_test_vec, emb_unlab_vec, train_lab, val_lab, test_lab, word_to_idx)
    weights_mat = gloVeLoader('glove/glove.6B.50d.txt', word_to_idx)
    #wordEmbeddingClassificationPretrained(device, emb_train_vec, emb_val_vec, emb_test_vec, emb_unlab_vec, train_lab, val_lab, test_lab, word_to_idx, weights_mat)
    #vanillaRNN(device, emb_train_vec, emb_val_vec, emb_test_vec, emb_unlab_vec, train_lab, val_lab, test_lab, word_to_idx, weights_mat)
    lstm(device, emb_train_vec, emb_val_vec, emb_test_vec, emb_unlab_vec, train_lab, val_lab, test_lab, word_to_idx, weights_mat)
if __name__ == "__main__":
    main()