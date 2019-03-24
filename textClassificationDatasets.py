import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random

class BagOfWordDataset(Dataset):
    def __init__(self, vec, label, transforms=None):
        self.transforms = transforms
        self.vec = vec
        self.label = torch.tensor(label, dtype=torch.float).view(-1)
        self.count = len(label)

    def __getitem__(self, index):
        sample = (self.vec[index,:], self.label[index])
        return sample

    def __len__(self):
        return self.count

class WordEmbeddingLearnedDataset(Dataset):
    def __init__(self, vec, label, transforms=None):
        self.transforms = transforms
        self.vec = vec
        self.label = torch.tensor(label, dtype=torch.float).view(-1)
        self.count = len(label)

    def __getitem__(self, index):
        sample = (self.vec[index,:], self.label[index])
        return sample

    def __len__(self):
        return self.count


class WordEmbeddingPretrainedDataset(Dataset):
    def __init__(self, vec, label, transforms=None):
        self.transforms = transforms
        self.vec = vec
        self.label = torch.tensor(label, dtype=torch.float).view(-1)
        self.count = len(label)

    def __getitem__(self, index):
        sample = (self.vec[index,:], self.label[index])
        return sample

    def __len__(self):
        return self.count

class VanillaRNNDataset(Dataset):
    def __init__(self, vec, label, transforms=None):
        self.transforms = transforms
        self.vec = vec
        self.label = torch.tensor(label, dtype=torch.float).view(-1)
        self.count = len(label)

    def __getitem__(self, index):
        sample = (self.vec[index,:], self.label[index])
        return sample

    def __len__(self):
        return self.count

class LSTMDataset(Dataset):
    def __init__(self, vec, label, transforms=None):
        self.transforms = transforms
        self.vec = vec
        self.label = torch.tensor(label, dtype=torch.float).view(-1)
        self.count = len(label)

    def __getitem__(self, index):
        sample = (self.vec[index,:], self.label[index])
        return sample

    def __len__(self):
        return self.count