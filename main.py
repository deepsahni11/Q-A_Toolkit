from Models.dynamic_coattention_model import *
from Models.config import *
from train import *
import os

class Config(object):
    pass

config = Config()
config.data_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\"
config.word_embedding_size = 100
config.hidden_dim = 300
config.dropout_ratio = 0.15
config.max_context_length = 600
config.max_question_length = 30


#vector with zeros for unknown words
config.num_iterations = 2
config.maxout_pool_size=16

config.lr = 0.001
config.dropout_ratio = 0.15

config.max_grad_norm = 5.0
config.batch_size = 11
config.num_epochs = 2

# config.print_every = 100
# config.save_every = 50000000
# config.eval_every = 1000

# config.model_type = 'co-attention'
config.reg_lambda = 0.00007
config.names = ["train_context","train_question"]
config.print_every = 100

hidden_dim = 100
dropout_ratio = 0.2
maxout_pool_size=16
max_number_of_iterations = 5
with open(r"E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\glove_word_embeddings.pkl", "rb") as input_file:
    embedding_matrix = pickle.load(input_file)

with autograd.set_detect_anomaly(True):
    model = DCN_Model(hidden_dim, embedding_matrix, dropout_ratio, maxout_pool_size, max_number_of_iterations)
    # config = Config()
    # model = model.cpu()
    train_model = Train_Model(config, model)

    train_model.train()
