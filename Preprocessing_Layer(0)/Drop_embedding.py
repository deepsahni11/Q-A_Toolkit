import numpy as np
import os
import tqdm as tqdm
import pickle

import numpy as np
import os
import spacy
# import ujson as json
import urllib.request

# from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile



class Embedding_Matrix():
    def __init__(self,embedding_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\DROP"):
#         embedding_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD"
        with open(r"E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\dictionaries.pkl", "rb") as input_file:
            dictionaries = pickle.load(input_file)
        self.word_to_index = dictionaries["word"]
        self.char_to_index = dictionaries["char"]
    
    def encode_word(self,word):
        if word in self.word_to_index:
            return self.word_to_index[word]
        else:
            return self.word_to_index["<unk>"]

    def encode_char(self,index):
        if index in self.char_to_index:
            return self.char_to_index[index]
        else:
            return "<unk>"
    
    def encodewords(self,typ,dataset):
    lines=""
    file_pat  = os.path.join(data_path, dataset+'_'+ty+'.txt')
    read_file = codecs.open(file_pat, "r", encoding="utf-8")
    lines =read_file.read()
    lines=lines.split("\n")
    en=[]
    for line in lines:
        w=""
        line=line.lower()
        words=line.split()
        for word in words:
            a=encode_word(word)
            w=w+str(a)+" "
        en.append(w)
        #print(en)
    cPickle.dump(en, open(os.path.join(data_path, dataset+"_encode_word_"+typ+".pkl"), "wb"))
    
    def encodechars(self,typ,dataset):
    lines=""
    file_pat  = os.path.join(data_path, dataset+'_'+typ+'.txt')
    read_file = codecs.open(file_pat, "r", encoding="utf-8")
    lines =read_file.read()
    lines=lines.split("\n")
    en=[]
    for line in lines:
        w=""
        line=line.lower()
        words=line.split()
        for word in words:
            for c in word:
                a=encode_char(c)
                w=w+str(a)+" "
            en.append(w)
        #print(en)
    cPickle.dump(en, open(os.path.join(data_path, dataset+"_encode_char_"+typ+".pkl"), "wb"))
    

    def get_glove_embeddings(self, word_embedding_size = 100, char_embedding_size = 20 , embedding_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD" ):



        glove_embeddings = os.path.join(embedding_dir, "glove.6B.100d.txt")

        glove_embeddings = open(glove_embeddings,'r', encoding = 'utf-8')

        temp_embeddings = []

        for word in self.word_to_index:

                if word in ['<pad>', '<s>', '<eos>']:
                    temp_vector = np.zeros((word_embedding_size))
                elif word not in glove_embeddings:
                    temp_vector = np.random.uniform(-np.sqrt(3)/np.sqrt(word_embedding_size), np.sqrt(3)/np.sqrt(word_embedding_size), word_embedding_size)
                else:
                    temp_vector = glove_embeddings[word]

                temp_embeddings.append(temp_vector)

        temp_embeddings = np.asarray(temp_embeddings)
        temp_embeddings = temp_embeddings.astype(np.float32)
        self.word_embeddings = temp_embeddings


        char_embeddings = []
        print (char_embedding_size)
        char_embeddings.append(np.zeros((char_embedding_size)))

        for i in range(len(self.char_to_index)):
            temp_vector = np.random.uniform(-np.sqrt(3)/np.sqrt(char_embedding_size), np.sqrt(3)/np.sqrt(char_embedding_size), char_embedding_size)
            char_embeddings.append(temp_vector)

        char_embeddings = np.asarray(char_embeddings)
        char_embeddings = char_embeddings.astype(np.float32)

        self.char_embeddings = char_embeddings

        pickle.dump(char_embeddings, open(os.path.join(embedding_dir, "char_embeddings" + ".pkl"), "wb")) 
        pickle.dump(temp_embeddings, open(os.path.join(embedding_dir, "glove_word_embeddings" + ".pkl"), "wb"))

