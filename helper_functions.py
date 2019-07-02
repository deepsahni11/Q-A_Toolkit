from __future__ import print_function

import json
import re
import string
import sys
from collections import Counter

import tqdm as tqdm
import torch
import random
# import nltk
import numpy as np
import pickle
import sys
import copy
import os.path
# import tqdm as tqdm

datapath = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\"

def find_max_length(data):

    """ Finds the maximum sequence length for data 
        Args:
            data: The data from which sequences will be chosen
    """
    temp = 0
    index = 0
    for i, _ in enumerate(data):

        if (len(data[i]) > temp):
            temp = len(data[i])
            index = i
    return temp,index


def pad_data(data):

    """ Pad the data to max_length given
        Args: 
            data: Data that needs to be padded
            max_length : The length to be achieved with padding
        Returns:
            padded_data : Each sequence is padded to make it of length
                          max_length.
    """
    padded_data = []
    max_length,index =  find_max_length(data)

    for lines in data:
        if (len(lines) < max_length):
            temp = np.lib.pad(lines, (0,max_length - len(lines)),
                'constant', constant_values=0)
        else:
            temp = lines[:max_length]
        padded_data.append(temp)

    padded_data = torch.from_numpy(np.array(padded_data)).type(torch.int64)

    return padded_data


def index_files_using_word_to_index(filename, _dict, max_words):
    
    f = open(filename, "r", encoding="utf-8")

    lines = f.readlines()
    lines  = [l.lower() for l in lines]
    encoded_lines = []
    for l in lines:
        tokens = l.split()
        tokens = tokens[:max_words]
        temp = []
        for t in tokens:
            if t in _dict:
                temp.append(_dict[t])
            else:
                temp.append(1)

        encoded_lines.append(temp[:])

    return encoded_lines




def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))
    
with open(r"E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\dictionaries.pkl", "rb") as input_file:
    dictionaries = pickle.load(input_file)
word_to_index = dictionaries["word_to_index"]
