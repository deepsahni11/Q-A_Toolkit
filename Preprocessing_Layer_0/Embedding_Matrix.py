
import numpy as np
import os
import tqdm as tqdm
import pickle

import numpy as np
import os
import spacy
# import ujson as json
import urllib.request
import numpy as np
# from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile



class Embedding_Matrix():

    def __init__(self,embedding_dir):
#         embedding_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD"
        with open(embedding_dir + "dictionaries.pkl", "rb") as input_file:
            dictionaries = pickle.load(input_file)
        self.word_to_index = dictionaries["word_to_index"]
        self.char_to_index = dictionaries["char_to_index"]
        self.index_to_word = dictionaries["index_to_word"]
        self.index_to_char = dictionaries["index_to_char"]


    def index_files_using_char_to_index(self, filename, _dict, max_words, max_chars):

        f = open(filename, "r", encoding="utf-8")
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
#                     print(_dict[t])
                    temp.append(_dict[t])
                else:
                    temp.append(_dict["<unk>"])

            encoded_lines.append(temp[:])
#         close(filename)
#             print("HEllo")

        return encoded_lines

    def index_files_to_char_level_and_word_level(self, datapath , max_words, max_chars):
#         files = [".context", ".question", ".answer_text"]
        files = [".context",".question", ".answer_text"]


        for f in files:
            read_path_train = os.path.join(datapath, "train" + f)
            write_path_train_word = os.path.join(datapath, "train_word_index" + f + "_pkl.pkl")
#             write_path_train_char = os.path.join(datapath, "train_char_index" + f + "_pkl.pkl")

            read_path_valid = os.path.join(datapath, "validation" + f)
            write_path_valid_word = os.path.join(datapath, "validation_word_index" + f + "_pkl.pkl")
#             write_path_valid_char = os.path.join(datapath, "validation_char_index" + f + "_pkl.pkl")



            temp_train_word = self.index_files_using_word_to_index(read_path_train, self.word_to_index, max_words)
            temp_valid_word = self.index_files_using_word_to_index(read_path_valid, self.word_to_index, max_words)

#             temp_train_char = index_files_using_char_to_index(read_path_train, self.char_to_index, max_words,max_chars)
#             temp_valid_char = index_files_using_char_to_index(read_path_valid, self.char_to_index, max_words,max_chars)

            write_file_train_word = open(write_path_train_word, "wb")
            pickle.dump(temp_train_word, write_file_train_word)

#             write_file_train_char = open(write_path_train_char, "wb")
#             pickle.dump(temp_train_char, write_file_train_char)

            write_file_valid_word = open(write_path_valid_word, "wb")
            pickle.dump(temp_valid_word, write_file_valid_word)

#             write_file_valid_char = open(write_path_valid_char, "wb")
#             pickle.dump(temp_valid_char, write_file_valid_char)

    def get_glove_embeddings(self, word_embedding_size , char_embedding_size  , embedding_dir  ):



        glove_embeddings = os.path.join(embedding_dir, "glove_embeddings100.txt")

        glove_embeddings = open(glove_embeddings,'r', encoding = 'utf-8')



        #     glove_embeddings = pickle.load(open(glove_embeddings))

        #####################  CHECK HOW GLOVE EMBEDDINGS WORK ##############
        temp_embeddings = []

        for word in self.word_to_index:

                if word in ['<pad>', '<sos>']:
                    temp_vector = np.zeros((word_embedding_size))
                elif word not in glove_embeddings:
                    temp_vector = np.random.uniform(-np.sqrt(3)/np.sqrt(word_embedding_size), np.sqrt(3)/np.sqrt(word_embedding_size), word_embedding_size)
                else:
                    temp_vector = glove_embeddings[word]

                temp_embeddings.append(temp_vector)

        temp_embeddings = np.asarray(temp_embeddings)
        temp_embeddings = temp_embeddings.astype(np.float32)
        self.word_embeddings = temp_embeddings


#         char_embeddings = []
# #         print (char_embedding_size)
#         char_embeddings.append(np.zeros((char_embedding_size)))

#         for i in range(len(self.char_to_index)):
#             temp_vector = np.random.uniform(-np.sqrt(3)/np.sqrt(char_embedding_size), np.sqrt(3)/np.sqrt(char_embedding_size), char_embedding_size)
#             char_embeddings.append(temp_vector)

#         char_embeddings = np.asarray(char_embeddings)
#         char_embeddings = char_embeddings.astype(np.float32)

#         self.char_embeddings = char_embeddings

#         pickle.dump(char_embeddings, open(os.path.join(embedding_dir, "char_embeddings" + ".pkl"), "wb"))
        pickle.dump(temp_embeddings, open(os.path.join(embedding_dir, "glove_word_embeddings" + ".pkl"), "wb"))


#         return self.word_embeddings, self.char_embeddings
