import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

'''
Binary NLP classifiers with no hidden layers
'''

class BagOfWordClassifier(nn.Module):  # inheriting from nn.Module
    def __init__(self, vocab_size, hidden_size=1):
        super(BagOfWordClassifier, self).__init__()
        self.net = nn.Sequential(
        nn.Linear(vocab_size, hidden_size),
        nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class WordEmbeddingClassifierLearned(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size=1):
        super(WordEmbeddingClassifierLearned, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.linear = nn.Linear(embed_dim, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embed = self.embedding(x.long())
        pooled = embed.mean(dim=1)
        return self.sigmoid(self.linear(pooled))

class WordEmbeddingClassifierPretrained(nn.Module):
    def __init__(self, vocab_size, weights_mat, embed_dim=50, hidden_size=1):
        super(WordEmbeddingClassifierPretrained, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.load_state_dict({'weight': weights_mat})
        self.linear = nn.Linear(embed_dim, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embed = self.embedding(x.long())
        pooled = embed.mean(dim=1)
        return self.sigmoid(self.linear(pooled))

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, weights_mat, hidden_size, embed_dim=50, linear_hidden_size=1):
        super(VanillaRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.load_state_dict({'weight': weights_mat})
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size,linear_hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embed = self.embedding(x.long())# batch 100, x \in 15 * 100
        scores,_ = self.rnn(embed)# 100, 15 * hidden_size
        return self.sigmoid(self.linear(scores.mean(dim=1)))

class LSTM(nn.Module):
    def __init__(self, vocab_size, weights_mat, hidden_size, embed_dim=50, linear_hidden_size=1):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.load_state_dict({'weight': weights_mat})
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size,linear_hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embed = self.embedding(x.long())# batch 100, x \in 15 * 100
        scores,_ = self.lstm(embed)# 100, 15 * hidden_size
        return self.sigmoid(self.linear(scores.mean(dim=1)))

class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, output_cha=128, kernel=5, hidden_size=1, max_pooling=True):
        super(CNN, self).__init__()
        self.max_pooling = max_pooling
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(max_length, output_cha, kernel)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(output_cha, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embed = self.embedding(x.long())
        conv = self.conv1d(embed)
        pooled = None
        if self.max_pooling:
            pooled, _ = conv.max(dim=2)
        else:
            pooled = conv.mean(dim=2)
        return self.sigmoid(self.linear(self.relu(pooled)))