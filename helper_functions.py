import tqdm as tqdm
import torch
import random
# import nltk
import numpy as np
import pickle
import sys
import copy
import os.path
import tqdm as tqdm

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

    for lines in tqdm.tqdm(data):
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

    
with open(r"E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\dictionaries.pkl", "rb") as input_file:
    dictionaries = pickle.load(input_file)
word_to_index = dictionaries["word_to_index"]
