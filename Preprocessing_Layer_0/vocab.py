import os.path
import operator
import _pickle as cPickle
from nltk.tokenize import WhitespaceTokenizer 
#from gensim.models import Word2Vec, KeyedVectors
from collections import defaultdict
from math import sqrt
import numpy as np 
import codecs
import re
import string
import sys
from collections import Counter
from argparse import ArgumentParser
import tqdm

class Vocab:

    """ The initial vocab will be created based on the tokens
        stored in the files that will be sent to this function
    """
    def __init__(self, token_filenames,answer_files, vocab_freq = 0, vocab_size = 30000, data_path="D:/Downloads/SQuAD/"):
        self.vocab = {}
        #self.vocab_output_filename = vocab_output_filename
        self.vocab_input_files = token_filenames
        self.word_list = []
        self.word_to_index = {} # dictionary with keys as words and values as their corresponding index number
        self.char_to_index = {} # dictionary with keys as characters and values as their corresponding index number
        self.word_to_index["<pad>"] = 0
        self.word_to_index["<unk>"] = 1
        self.data_path = data_path
        
        for filename in self.vocab_input_files:
            with open(self.data_path + filename,'r', encoding = 'utf-8') as file_input:
                
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
        self.index_to_word =  {v: k for k, v in self.word_to_index.items()}


        characters = list(string.printable.lower())
        characters.remove(' ')

        char_ind = 1
        for c in characters:
            if c not in self.char_to_index:
                self.char_to_index[c] = char_ind
                char_ind += 1


        self.index_to_char = {v: k for k,v in self.char_to_index.items()}

        dict_all = {"word" : self.word_to_index, "char" : self.char_to_index,"in_word":self.index_to_word ,"in_char":self.index_to_char}
        
        self.word_to_index = dict_all["word"]
        self.char_to_index = dict_all["char"]
        self.index_to_word=dict_all["in_word"]
        self.index_to_char=dict_all["in_char"]
        
        cPickle.dump(dict_all, open(os.path.join(data_path, "dictionaries.pkl"), "wb"))
        
    

    def get_glove_embeddings(self, word_embedding_size = 100, char_embedding_size = 20 , embedding_dir = "D:/Downloads/SQuAD/" ):



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

        cPickle.dump(char_embeddings, open(os.path.join(embedding_dir, "char_embeddings" + ".pkl"), "wb")) 
        cPickle.dump(temp_embeddings, open(os.path.join(embedding_dir, "word_embeddings" + str(word_embedding_size) + ".pkl"), "wb"))
        return self.word_embeddings, self.char_embeddings
        
    def convert_file_to_tokens(self, filename, _dict, max_words):

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

    def convert_file_to_chars(self, filename, _dict, max_words, max_chars):

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
                l = l[:max_chars]
                encoded_chars = []
                for j in l:
                    if j in _dict:
                        encoded_chars.append(_dict[j])
                    else:
                        encoded_chars.append(0)
                encoded_tokens.append(encoded_chars)
            encoded_lines.append(encoded_tokens)

        return encoded_lines


    def convert_all_files(self, token_files,answer_files,datapath = "D:/Downloads/SQuAD/",max_words=0, max_chars=0):
        
        #token_files = ["train_context.txt_token", "train_question.txt_token","test_question.txt_token", "test_context.txt_token"]
        #answer_files = ["train_answer.txt_token","test_answer.txt_token"]
        #token_files = ["train_question.txt", "train_context.txt", "valid_question.txt", "valid_context.txt"]
        #pos_files  = ["train_query_pos.txt", "train_context_pos.txt", "valid_query_pos.txt", "valid_context_pos.txt"]
        #ner_files  = ["train_query_ner.txt", "train_context_ner.txt", "valid_context_ner.txt", "valid_query_ner.txt"]
        #context_words = ["train_context_token.txt", "valid_context_token.txt"]
        #answer_files = ["valid_answer.txt","train_answer.txt"]
        #ground_truth_files = ["train_ground_truths.txt", "valid_ground_truths.txt"]
        
        for f in answer_files:
            read_path = os.path.join(datapath, f)
            write_path = os.path.join(datapath, f + ".txt_pkl.pkl")
            lines = open(read_path, "r",encoding='utf-8').readlines()
            temp = []
            for l in lines:
                if "text" not in read_path:
                    temp.append([int(l)])
                else:
                    temp.append([l])
                
            if "train" in write_path:
                train = temp[7602:]
                valid = temp[:7602]
                write_file = open(write_path, "wb")
                cPickle.dump(train, write_file)
                valid_path = write_path.replace("train", "validation")
                valid_file = open(valid_path, "wb")
                cPickle.dump(valid, valid_file)
            else:
                test_path = write_path.replace("validation", "test")
                test_file = open(test_path, "wb")
                cPickle.dump(temp, test_file)
        
        
        for f in token_files:
            read_path = os.path.join(datapath, f)
            write_path = os.path.join(datapath, f + "_pkl.pkl")
            temp = self.convert_file_to_tokens(read_path, self.word_to_index, max_words)
            if "train" in write_path:
                valid = temp[:7602]
                train = temp[7602:]
                write_file = open(write_path, "wb")
                cPickle.dump(train, write_file)
                valid_path = write_path.replace("train", "validation")
                valid_file = open(valid_path, "wb")
                cPickle.dump(valid, valid_file)
            else:
                test_path = write_path.replace("validation", "test")
                test_file = open(test_path, "wb")
                cPickle.dump(temp, test_file)
                #print(train)
        
        for f in token_files:
            read_path  = os.path.join(datapath, f)
            write_path = os.path.join(datapath, f + "_char_pkl.pkl")
            
            temp = self.convert_file_to_chars(read_path, self.char_to_index, max_words, max_chars)

            if "train" in write_path:
                valid = temp[:7602]
                train = temp[7602:]
                
                write_file = open(write_path, "wb")
                cPickle.dump(train, write_file)
                valid_path = write_path.replace("train", "validation")
                valid_file = open(valid_path, "wb")
                cPickle.dump(valid, valid_file)
            else:
                test_path = write_path.replace("validation", "test")
                test_file = open(test_path, "wb")
                cPickle.dump(temp, test_file)



def main():

    #token_filenames = ["train_query", "valid_query", "test_query", "train_context", "valid_context", "test_context"]
    #pos_filenames = ["train_query_pos", "valid_query_pos", "test_query_pos", "train_context_pos", "valid_context_pos", "test_context_pos"]
    #ner_filenames = ["train_query_ner", "valid_query_ner", "test_query_ner", "train_context_ner", "valid_context_ner", "test_context_ner"]

    token_filenames = [ "train.question","train.context","validation.question", "validation.context"]
    answer_files = ["train.answer_text","train.answer_start","train.answer_end","validation.answer_text","validation.answer_end","validation.answer_start"]
    
    
    #pos_filenames  = ["train_query_pos.txt", "train_context_pos.txt"]
    #ner_filenames  = ["train_query_ner.txt", "train_context_ner.txt"]

    vocab = Vocab(token_filenames, answer_files, vocab_freq = 0, vocab_size = 20000, data_path = "D:/Downloads/SQuAD/")
    
    vocab.get_glove_embeddings(word_embedding_size = 100, char_embedding_size = 8, embedding_dir = "D:/Downloads/SQuAD/")
    
   
    vocab.convert_all_files( token_filenames,answer_files,"D:/Downloads/SQuAD/",10000, 15)

if __name__ == '__main__':
    flags = ArgumentParser(description='Model Tester')
    flags.add_argument("--char_emb_size", type=int, default=8, help="Character embedding size")
    flags.add_argument("--word_emb_size", type=int, default=100, help="Word embedding size")
    flags.add_argument("--vocab_size", type=int, default=20000, help="Vocab size")
    flags.add_argument("--vocab_freq", type=int, default=0, help = "Vocab frequency")

    flags.add_argument("--convert_tokens", type=bool, default= True, help="convert tokens")
    flags.add_argument("--get_pretrained_embedding", type=bool, default= True, help="get pretrained embedding")

    flags.add_argument("--data_dir", type=str, default="../data", help="Data path")
    flags.add_argument("--embedding_dir", type=str, default="../data", help="Embedding directory")

    flags.add_argument("--max_words", type=int, default=1000, help= "maximum number of words per sentence")
    flags.add_argument("--max_chars", type=int, default=15, help="max number of character per sentence")

    #config = flags.parse_args()

    main()
