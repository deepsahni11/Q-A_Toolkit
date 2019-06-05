
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
    def __init__(self, vocab_input_files = ["E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\train.context","E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\train.question"],
                        vocab_output_filename = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\vocab.dat"):
        self.vocab = {}
        self.vocab_output_filename = vocab_output_filename
        self.vocab_input_files = vocab_input_files
        self.word_list = []
        self.word_to_index = {} # dictionary with keys as words and values as their corresponding index number
        self.char_to_index = {}
        self.word_to_index["<pad>"] = 0
        self.word_to_index["<unk>"] = 1
        
    def create_vocabulary(self,vocab_freq = 0, vocab_size = 30000, data_path="E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD"):
        for filename in self.vocab_input_files:
            with open(filename,'r', encoding = 'utf-8') as file_input:
                words = " "
                for line in tqdm(file_input):
                    words = words + " " + line.strip().split() # creates a list of words per line
                            
                temp_vocab = Counter(words)
                if vocab_freq == 0:
                    vocab_words = temp_vocab.most_common(vocab_size)

                else:
                    vocab_words = []
                    for i in temp_vocab:
                        if (temp_vocab[i] > vocab_freq):
                            vocab_words.append([i, temp_vocab[i]])

                    print ("Vocab size is ", len(vocab_words))
                    
                temp_index = 2
                for word in vocab_words:
                    if word[0] not in self.word_to_index:
                        self.word_to_index[word[0]] = temp_index
                        temp_index += 1

                self.vocab_size = len(self.word_to_index)
                self.index_to_word =  {v: k for k, v in self.word_to_index.iteritems()}

                
                characters = list(string.printable.lower())
                characters.remove(' ')

                char_ind = 1
                for c in characters:
                    if c not in self.char_to_index:
                        self.char_to_index[c] = char_ind
                        char_ind += 1

                
                self.index_to_char = {v: k for k,v in self.char_to_index.iteritems()}
                
                dict_all = {"word" : self.word_to_index, "char" : self.char_to_index}

                pickle.dump(dict_all, open(os.path.join(data_path, "dictionaries.pkl"), "wb")) ## creates dictionaries and stores in memory as pickle files
    def index_files_using_char_to_index(self, filename, _dict, max_words, max_chars):
        f = codecs.open(filename, "r", encoding="utf-8")
        lines = f.readlines()
        lines = [l.lower() for l in lines]
        encoded_lines = []
        for l in lines:
            tokens = l.split()
            tokens = tokens[:max_words]
            encoded_tokens = []
            for t in tokens:
                l = list(t)
                l = l[:max_chars] ## there is a max limit for the length of characters = max_chars
                encoded_chars = []
                for j in l:
                    if j in _dict:
                        encoded_chars.append(_dict[j])
                    else:
                        encoded_chars.append(0)  ## if the character id not in dictionary put '0' in its place
                encoded_tokens.append(encoded_chars)
            encoded_lines.append(encoded_tokens)

        return encoded_lines
        
    def index_files_using_word_to_index(self, filename, _dict, max_words):
        f = codecs.open(filename, "r", encoding="utf-8")

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
    def index_files_to_char_level_and_word_level(self, datapath = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD", max_words=0, max_chars=0):
        files = [".context", ".query", ".answer_text"]

        for f in files:
            read_path_train = os.path.join(datapath, "train" + f)
            write_path_train_word = os.path.join(datapath, "train_word_index" + f + "_pkl.pkl")
            write_path_train_char = os.path.join(datapath, "train_char_index" + f + "_pkl.pkl")
              
            read_path_valid = os.path.join(datapath, "validation" + f)
            write_path_valid_word = os.path.join(datapath, "validation_word_index" + f + "_pkl.pkl")
            write_path_valid_char = os.path.join(datapath, "validation_char_index" + f + "_pkl.pkl")
            
            

            temp_train_word = index_files_using_word_to_index(self, read_path_train, self.word_to_index, max_words)
            temp_valid_word = index_files_using_word_to_index(self, read_path_valid, self.word_to_index, max_words)
            
            temp_train_char = index_files_using_char_to_index(self, read_path_train, self.char_to_index, max_words,max_chars)
            temp_valid_char = index_files_using_char_to_index(self, read_path_valid, self.char_to_index, max_words,max_chars)
        
            write_file_train_word = open(write_path_train_word, "wb")
            pickle.dump(temp_train_word, write_file_train_word)
            
            write_file_train_char = open(write_path_train_char, "wb")
            pickle.dump(temp_train_char, write_file_train_char)
            
            write_file_valid_word = open(write_path_valid_word, "wb")
            pickle.dump(temp_valid_word, write_file_valid)
            
            write_file_valid_char = open(write_path_valid_char, "wb")
            pickle.dump(temp_valid_char, write_file_valid)
            
    def get_pretrained_embeddings(self, word_embedding_size = 100, char_embedding_size = 20 , embedding_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\embeddings" ):
        
        glove_embeddings = os.path.join(embedding_dir, "glove_embeddings100.txt")
        glove_embeddings = open(glove_embeddings,'r', encoding = 'utf-8')
        glove_embeddings = pickle.load(open(glove_embeddings))
        
        #####################  CHECK HOW GLOVE EMBEDDINGS WORK ##############
        temp_embeddings = []

        for word in self.word_to_index:

                if word in ['<pad>', '<s>', '<eos>']:
                    temp_vector = np.zeros((word_embedding_size))
                elif word not in glove_embeddings:
                    temp_vector = np.random.uniform(-sqrt(3)/sqrt(word_embedding_size), sqrt(3)/sqrt(word_embedding_size), word_embedding_size)
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
            temp_vector = np.random.uniform(-sqrt(3)/sqrt(char_embedding_size), sqrt(3)/sqrt(char_embedding_size), char_embedding_size)
            char_embeddings.append(temp_vector)

        char_embeddings = np.asarray(char_embeddings)
        char_embeddings = char_embeddings.astype(np.float32)

        self.char_embeddings = char_embeddings

        pickle.dump(char_embeddings, open(os.path.join(embedding_dir, "char_embeddings" + str(char_embedding_size) + ".pkl"), "wb")) 
        pickle.dump(temp_embeddings, open(os.path.join(embedding_dir, "word_embeddings" + str(word_embedding_size) + ".pkl"), "wb"))
        

        return self.word_embeddings, self.char_embeddings





        
def main(config):
    vocab = Vocabulary(self, ["E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\train.context","E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\train.question"],\
                        "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\vocab.dat")
    
    if config.get_pretrained_embedding:
        vocab.get_pretrained_embeddings(word_embedding_size = config.word_emb_size, char_embedding_size = config.char_emb_size, embedding_dir = config.embedding_dir)

    if config.convert_tokens:
        vocab.index_files_to_char_level_and_word_level(config.data_dir, config.max_words, config.max_chars)

    

if __name__ == '__main__':
    ##### Add all the relevant flags
    
    flags = ArgumentParser(description = "Vocabulary Building and Preprocessing Stage") ## makes an object of class ArgumentParser
    flags.add_argument("--char_emb_size", type=int, default=8, help="Character embedding size")
    flags.add_argument("--word_emb_size", type=int, default=100, help="Word embedding size")
    flags.add_argument("--vocab_size", type=int, default=20000, help="Vocabulary size")
    flags.add_argument("--vocab_freq", type=int, default=0, help = "Vocabulary frequency")
    flags.add_argument("--convert_tokens", type=bool, default= True, help="convert tokens")
    flags.add_argument("--get_pretrained_embedding", type=bool, default= True, help="get pretrained embedding")
    flags.add_argument("--data_dir", type=str, default="E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD", help="Data path")
    flags.add_argument("--embedding_dir", type=str, default="E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD", help="Embedding directory")
    flags.add_argument("--max_words", type=int, default=1000, help= "maximum number of words per sentence")
    flags.add_argument("--max_chars", type=int, default=15, help="max number of character per sentence")

    config = flags.parse_args()

    main(config)
    
