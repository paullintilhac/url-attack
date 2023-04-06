import time
import math
import random
import string
from collections import defaultdict
from random import choice, shuffle
from functools import reduce
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from numpy import dot
from numpy.linalg import norm

import re
from pythomata import SimpleDFA
import graphviz
import matplotlib.pyplot as plt 

from sklearn.mixture import BayesianGaussianMixture
from sklearn.manifold import LocallyLinearEmbedding, locally_linear_embedding
from sklearn.neighbors import NearestNeighbors

import gudhi as gd
from gudhi.clustering.tomato import Tomato
direct = "."

tomita_dict = {tomita_grammar: direct + f'/tomita{tomita_grammar}.txt'
               for tomita_grammar in [-3, 1, 2, 3, 4, 5, 6, 7]}
import time
import math
import random

def tokenized_dict(alphabet):
    return dict((c, i) for i, c in enumerate(alphabet))

def seq_to_tokens(word, lookup_dict: dict):
    print("word: " + str(word))
    return [lookup_dict[letter] for letter in word] if isinstance(word, (list, tuple)) else lookup_dict[
        word] if word is not None else []


def split_train_validation(x_values, y_values, ratio=0.8, uniform=False):

    num_cat = len(set(y_values))
    x_train, y_train, x_test, y_test = None, None, None, None
    target_cat = 0
    # both train and test should have all labels
    while True:
        c = list(zip(x_values, y_values))
        
        shuffle(c)
        x_values, y_values = zip(*c)
        cutoff = int((len(x_values) + 1) * ratio)
        x_train, x_test = x_values[:cutoff], x_values[cutoff:]
        y_train, y_test = y_values[:cutoff], y_values[cutoff:]
        if not uniform and len(set(y_train)) == num_cat:  # and len(set(y_test)) == num_cat:
            break
        elif len(set(y_train)) == num_cat and len(set(y_test)) == num_cat:
            break
    
    return x_train, y_train, x_test, y_test

def letterToTensor(letter):
    tensor = np.zeros((1, 2))
    tensor[0][letter] = 1
    return tensor

def lineToTensor(line,max_len,n_letters=2):
    tensor = torch.zeros(max_len,n_letters)
    startIndex = max_len-len(line)
    for li, letter in enumerate(line):
        tensor[li+startIndex][letter] = 1
    return tensor

def categoryToTensor(category,n_letters=2):
    tensor = torch.zeros(n_letters)
    tensor[category] = 1
    return tensor

def preprocess_binary_classification_data(x, y, alphabet):
    char_dict = tokenized_dict(alphabet)
    tokenized_x = [seq_to_tokens(word, char_dict) for word in x]
    print("tokenized_x[10]: " + str(tokenized_x[0]))
    print("tokenized_x[11]: " + str(tokenized_x[1]))

    maxL = len(max(tokenized_x, key=len))
    padded_x = []
    for x in tokenized_x:
        # print(lineToTensor(x,max_len=maxL).size())
        # print(lineToTensor(x,max_len=maxL).size(0))
        padded_x.append(lineToTensor(x,max_len=maxL))
    
    tokenized_y = [1 if i == 'True' else 0 for i in y]
    x_train, y_train, x_test, y_test = split_train_validation(padded_x, tokenized_y)
    return x_train, y_train, x_test, y_test


def parse_data(path: str):
    x, y = [], []
    file = open(path, 'r')
    lines = file.readlines()
    for l in lines:
        key_val_pair = l.rstrip().split(':')
        x.append(key_val_pair[0])
        y.append(key_val_pair[1])
    file.close()
    return x, y

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return input_alphabet[category_i], category_i
  


def parse_data(path: str):
    x, y = [], []
    file = open(path, 'r')
    lines = file.readlines()
    for l in lines:
        key_val_pair = l.rstrip().split(':')
        x.append(key_val_pair[0])
        y.append(key_val_pair[1])
    file.close()
    return x, y

direct = "."

tomita_dict = {tomita_grammar: direct + f'/tomita{tomita_grammar}.txt'
               for tomita_grammar in [-3, 1, 2, 3, 4, 5, 6, 7]}

tomita_grammar = 6
input_alphabet = ["0", "1"]
vocab_size = len(input_alphabet) + 1
input_dim = vocab_size - 1
path = tomita_dict[tomita_grammar]
x, y = parse_data(path)
x_train, y_train, x_test, y_test = split_train_validation(x, y)
print("x_train: " + str(x_train))