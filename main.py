from Models.dynamic_coattention_model import *
from Preprocessing_Layer_0.Vocabulary_builder import *
from Preprocessing_Layer_0.Squad_processor import *
from Preprocessing_Layer_0.Embedding_Matrix import *

from Models.config import *
from train import *
import os

class Config(object):
    pass

config = Config()
config.data_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\"
config.model_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\"
config.word_embedding_size = 100
config.hidden_dim = 100
# config.dropout_ratio = 0.15
config.max_context_length = 600
config.max_question_length = 30
config.word_embedding_size = 100
config.char_embedding_size = 20
config.max_words=700
config.lr = 0.001
config.dropout_ratio = 0.15
config.early_stop = 10
config.vocab_size = 50000


#vector with zeros for unknown words
config.num_iterations = 2
config.maxout_pool_size=16
config.max_chars=10

config.lr = 0.001
config.dropout_ratio = 0.15

config.max_grad_norm = 5.0
config.batch_size = 20
config.num_epochs = 2
config.print_every = 100
config.max_number_of_iterations = 5
config.print_and_validate_every = 2
# config.print_every = 100
# config.save_every = 50000000
# config.eval_every = 1000

# config.model_type = 'co-attention'
config.reg_lambda = 0.00007
config.names = ["train_context","train_question"]
config.print_every = 100
# config.vocab_size = 30000

# hidden_dim = 100
# dropout_ratio = 0.2
# maxout_pool_size=16
# max_number_of_iterations = 5

with open(r"E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\glove_word_embeddings.pkl", "rb") as input_file:
    embedding_matrix = pickle.load(input_file)

preprocess = Squad_preprocessor(nltk.word_tokenize,"E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD")
preprocess.conduct_preprocess()

data_directory = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD"
context_file = open(os.path.join(data_directory, 'train.context'), 'r', encoding='utf-8').readlines()
question_file = open(os.path.join(data_directory, 'train.question'), 'r', encoding='utf-8').readlines()
answer_text_file = open(os.path.join(data_directory,  'train.answer_text'), 'r', encoding= 'utf-8').readlines()
answer_start_file = open(os.path.join(data_directory,  'train.answer_start'), 'r', encoding= 'utf-8').readlines()
answer_end_file = open(os.path.join(data_directory,  'train.answer_end'), 'r', encoding= 'utf-8').readlines()

with open(os.path.join(data_directory, "validation.context" ), 'w', encoding='utf-8') as f:
    for item in context_file[8000:]:
        f.write("%s" % item)
with open(os.path.join(data_directory, "validation.question" ), 'w', encoding='utf-8') as f:
    for item in question_file[8000:]:
        f.write("%s" % item)
with open(os.path.join(data_directory, "validation.answer_text" ), 'w', encoding='utf-8') as f:
    for item in answer_text_file[8000:]:
        f.write("%s" % item)
with open(os.path.join(data_directory, "validation.answer_start" ), 'w', encoding='utf-8') as f:
    for item in answer_start_file[8000:]:
        f.write("%s" % item)
with open(os.path.join(data_directory, "validation.answer_end" ), 'w', encoding='utf-8') as f:
    for item in answer_end_file[8000:]:
        f.write("%s" % item)
with open(os.path.join(data_directory, "train.context"), 'w', encoding='utf-8') as f:
    for item in context_file[:8000]:
        f.write("%s" % item)
with open(os.path.join(data_directory, "train.question" ), 'w', encoding='utf-8') as f:
    for item in question_file[:8000]:
        f.write("%s" % item)
with open(os.path.join(data_directory, "train.answer_text" ), 'w', encoding='utf-8') as f:
    for item in answer_text_file[:8000]:
        f.write("%s" % item)
with open(os.path.join(data_directory, "train.answer_start"), 'w', encoding='utf-8') as f:
    for item in answer_start_file[:8000]:
        f.write("%s" % item)
with open(os.path.join(data_directory, "train.answer_end" ), 'w', encoding='utf-8') as f:
    for item in answer_end_file[:8000]:
        f.write("%s" % item)


vocab = Vocabulary(["E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\train.context","E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\train.question"],
                        "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dataset_analysis\\SQuAD\\vocab.dat")
vocab.create_vocabulary(0,config.vocab_size, config.data_dir)

embedding = Embedding_Matrix(config.data_dir)
embedding.get_glove_embeddings(config.word_embedding_size, config.char_embedding_size ,config.data_dir )
embedding.index_files_to_char_level_and_word_level(config.data_dir , config.max_words, config.max_chars)


with autograd.set_detect_anomaly(True):
    model = DCN_Model(config.hidden_dim, embedding_matrix, config.dropout_ratio, config.maxout_pool_size, config.max_number_of_iterations)
    # config = Config()
    # model = model.cpu()
    train_model = Train_Model(config, model)

    train_model.train()
