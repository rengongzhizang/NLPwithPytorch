import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from textClassificationDatasets import *
from textClassifiers import *

def make_words_dict(train_data, val_data, test_data, unlabelled_data):
    word_to_ix = {}
    for lines in train_data + val_data + test_data + unlabelled_data:
        for word in lines:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros((len(sentence), len(word_to_ix)))
    for i,line in enumerate(sentence):
        for word in line:
            vec[i][word_to_ix[word]] += 1
    return vec

def make_embed_vector(sentence, word_to_ix):
    max_length = 0
    for line in sentence:
        max_length = max(max_length, len(line))

    vec = torch.zeros((len(sentence), max_length))
    for i,line in enumerate(sentence):
        for j,word in enumerate(line):
            vec[i][j] = word_to_ix[word]
    return vec

def fileLoader(path, with_label=True):
    f = open(path, 'r')
    sentences = []
    labels = []
    if with_label == True:
        for line in f:
            labels.append(int(line[0]))
            sentences.append(line[2:].split())
        return sentences, labels
    else:
        for line in f:
            sentences.append(line[2:].split())
        return sentences

def gloVeLoader(path, word_to_ix, dim=50):
    f = open(path, 'r')
    embed_gloVe_dict = {}
    weights_mat = torch.zeros(len(word_to_ix), dim)
    for line in f:
        splitedLine = line.split()
        embed_gloVe_dict[splitedLine[0]] = splitedLine[1:]
    for word in word_to_ix:
        if word in embed_gloVe_dict:
            weights_mat[word_to_ix[word]] = torch.tensor([float(x) for x in embed_gloVe_dict[word]])
    return weights_mat

def train_eval(device, net, loss_fn, optimizer, train_loader, val_loader, epoch_num=10):
    print('Start Training')
    best_params = 0.0
    best_acc = 0.0
    for epoch in range(epoch_num):
        print('Epoch Num: {} / {} \n -------------------------'.format(epoch, epoch_num))
        net.train()
        running_loss = 0.0
        running_acc = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores = net(inputs)
            loss = loss_fn(scores, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_acc += torch.sum(torch.tensor((torch.round(scores.view(-1)) == labels.data)))
            #tmp = torch.sum(torch.tensor((torch.round(scores.view(-1)) == labels.data)))
            #temp = inputs.size(0)
            #print(running_loss, running_acc)
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc.double() / len(train_loader)
        print('Training loss: {:.4f}, Training accuracy: {:.4f}'.format(epoch_loss, epoch_acc))

        net.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores = net(inputs)
            loss = loss_fn(scores, labels.float())
            optimizer.zero_grad()
            val_running_loss += loss.item() * inputs.size(0)
            val_running_acc += torch.sum(torch.tensor((torch.round(scores.view(-1)) == labels.data)))

        val_loss = val_running_loss / len(val_loader)
        val_acc = val_running_acc.double() / len(val_loader)
        print('Validation loss: {:.4f}, Validation accuracy: {:.4f}'.format(val_loss, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = net.state_dict()
    net.load_state_dict(best_params)
    return net

def test(device, net, test_loader):
    net.eval()
    test_running_acc = 0.0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores = net(inputs)
        test_running_acc += torch.sum(torch.tensor((torch.round(scores.view(-1)) == labels.data)))

    test_acc = test_running_acc.double() / len(test_loader)
    print('Test accuracy: {:.4f}'.format(test_acc))

def generator(device, net, loader, path):
    file = open(path,'w')
    net.eval()
    for inputs, _ in loader:
        inputs = inputs.to(device)
        scores = net(inputs)
        outputs = torch.round(scores.view(-1))
        for num in outputs:
            file.write(str(int(num)) + '\n')

def bagOfWordClassification(device, train_vec, val_vec, test_vec, unlab_vec, train_lab, val_lab, test_lab): #path
    train_data = BagOfWordDataset(train_vec, train_lab)
    val_data = BagOfWordDataset(val_vec, val_lab)
    test_data = BagOfWordDataset(test_vec, test_lab)
    unlab_data = BagOfWordDataset(unlab_vec, [0 for i in unlab_vec])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)
    unlab_loader = torch.utils.data.DataLoader(unlab_data, batch_size=100, shuffle=False)
    net = BagOfWordClassifier(train_vec.shape[1])
    loss_fn = nn.BCELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    trained_net = train_eval(device, net, loss_fn, optimizer, train_loader, val_loader)
    test(device, trained_net, test_loader)
    generator(device, trained_net, unlab_loader, 'data/bagOfWordPrediction.txt')

def wordEmbeddingClassificationLearned(device, train_vec, val_vec, test_vec, unlab_vec, train_lab, val_lab, test_lab, word_to_ix):
    train_data = WordEmbeddingLearnedDataset(train_vec, train_lab)
    val_data = WordEmbeddingLearnedDataset(val_vec, val_lab)
    test_data = WordEmbeddingLearnedDataset(test_vec, test_lab)
    unlab_data = WordEmbeddingLearnedDataset(unlab_vec, [0 for i in unlab_vec])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)
    unlab_loader = torch.utils.data.DataLoader(unlab_data, batch_size=100, shuffle=False)
    net = WordEmbeddingClassifierLearned(len(word_to_ix), 20)
    loss_fn = nn.BCELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    trained_net = train_eval(device, net, loss_fn, optimizer, train_loader, val_loader)
    test(device, trained_net, test_loader)
    generator(device, trained_net, unlab_loader, 'data/wordEmbeddingClassificationLearnedPrediction.txt')

def wordEmbeddingClassificationPretrained(device, train_vec, val_vec, test_vec, unlab_vec, train_lab, val_lab, test_lab, word_to_ix, weights_mat):
    train_data = WordEmbeddingPretrainedDataset(train_vec, train_lab)
    val_data = WordEmbeddingPretrainedDataset(val_vec, val_lab)
    test_data = WordEmbeddingPretrainedDataset(test_vec, test_lab)
    unlab_data = WordEmbeddingPretrainedDataset(unlab_vec, [0 for i in unlab_vec])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)
    unlab_loader = torch.utils.data.DataLoader(unlab_data, batch_size=100, shuffle=False)
    net = WordEmbeddingClassifierPretrained(len(word_to_ix),weights_mat)
    loss_fn = nn.BCELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    trained_net = train_eval(device, net, loss_fn, optimizer, train_loader, val_loader)
    test(device, trained_net, test_loader)
    generator(device, trained_net, unlab_loader, 'data/wordEmbeddingClassificationPretrainedPrediction.txt')

def vanillaRNN(device, train_vec, val_vec, test_vec, unlab_vec, train_lab, val_lab, test_lab, word_to_ix, weights_mat):
    train_data = VanillaRNNDataset(train_vec, train_lab)
    val_data = VanillaRNNDataset(val_vec, val_lab)
    test_data = VanillaRNNDataset(test_vec, test_lab)
    unlab_data = VanillaRNNDataset(unlab_vec, [0 for i in unlab_vec])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)
    unlab_loader = torch.utils.data.DataLoader(unlab_data, batch_size=100, shuffle=False)
    net = VanillaRNN(len(word_to_ix), weights_mat, 10)
    loss_fn = nn.BCELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    trained_net = train_eval(device, net, loss_fn, optimizer, train_loader, val_loader)
    test(device, trained_net, test_loader)
    generator(device, trained_net, unlab_loader, 'data/vanillaRNNPrediction.txt')

def lstm(device, train_vec, val_vec, test_vec, unlab_vec, train_lab, val_lab, test_lab, word_to_ix, weights_mat):
    train_data = LSTMDataset(train_vec, train_lab)
    val_data = LSTMDataset(val_vec, val_lab)
    test_data = LSTMDataset(test_vec, test_lab)
    unlab_data = LSTMDataset(unlab_vec, [0 for i in unlab_vec])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)
    unlab_loader = torch.utils.data.DataLoader(unlab_data, batch_size=100, shuffle=False)
    net = LSTM(len(word_to_ix), weights_mat, 10)
    loss_fn = nn.BCELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)

    trained_net = train_eval(device, net, loss_fn, optimizer, train_loader, val_loader)
    test(device, trained_net, test_loader)
    generator(device, trained_net, unlab_loader, 'data/LSTMPrediction.txt')