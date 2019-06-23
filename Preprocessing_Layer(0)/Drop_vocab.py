import os.path
import operator
import pickle
from nltk.tokenize import WhitespaceTokenizer 
from gensim.models import Word2Vec, KeyedVectors
from collections import defaultdict
from math import sqrt
import numpy as np 
import codecs
import re
import string
import sys
import tqdm as tqdm
import os
from collections import Counter
from argparse import ArgumentParser


class Vocabulary():
    def __init__(self, vocab_input_files = ["E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\DROP\\train_context.txt","E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\DROP\\train_question.txt"],
       
        self.vocab = {}
        self.vocab_output_filename = vocab_output_filename
        self.vocab_input_files = vocab_input_files
        self.word_list = []
        self.word_to_index = {} # dictionary with keys as words and values as their corresponding index number
        self.char_to_index = {} # dictionary with keys as characters and values as their corresponding index number
        self.word_to_index["<pad>"] = 0
        self.word_to_index["<unk>"] = 1
        ## self.index_to_word = # dictionary with values as words and keys as their corresponding index number
        ## self.index_to_char = # dictionary with values as characters and keys as their corresponding index number
        
    
     
    
    
    def create_vocabulary(self,vocab_freq = 0, vocab_size = 30000, data_path="E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\DROP"):
        

        for filename in self.vocab_input_files:
            with open(filename,'r', encoding = 'utf-8') as file_input:
                
                for line in tqdm.tqdm(file_input):
                    words = line.strip().split()
#                     print(words)
                    for word in words:
                        if not (word in self.vocab):
                            self.vocab[word] = 1
                        else:
                            self.vocab[word] +=1 

        if vocab_freq == 0:
            vocab_words = sorted(self.vocab,key=self.vocab.get,reverse=True)


                    
        temp_index = 2
        for word in vocab_words:
            if word not in self.word_to_index:
                self.word_to_index[word] = temp_index
                temp_index += 1
                
#         print(len(self.word_to_index))

        self.vocab_size = len(self.word_to_index)
        #self.index_to_word =  {v: k for k, v in self.word_to_index.items()}


        characters = list(string.printable.lower())
        characters.remove(' ')

        char_ind = 1
        for c in characters:
            if c not in self.char_to_index:
                self.char_to_index[c] = char_ind
                char_ind += 1


        self.index_to_char = {v: k for k,v in self.char_to_index.items()}

        dict_all = {"word" : self.word_to_index, "char" : self.char_to_index,"index_to_word": self.index_to_word, "index_to_char": self.index_to_char}

        pickle.dump(dict_all, open(os.path.join(data_path, "dictionaries.pkl"), "wb")) ## creates dictionaries and stores in memory as pickle files

