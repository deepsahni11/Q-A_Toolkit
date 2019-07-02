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
config.print_and_validate_every = 2

#vector with zeros for unknown words
config.num_iterations = 2
config.maxout_pool_size=16

config.lr = 0.001
config.dropout_ratio = 0.15
config.early_stop = 10
config.vocab_size = 50000

config.max_grad_norm = 5.0
config.batch_size = 20
config.num_epochs = 2
config.model_dir = "E:\\Internships_19\\Internship(Summer_19)\\Q&A_Toolkit\\Dynamic_Coattention_Networks\\Models\\saved_models"

# config.print_every = 100
# config.save_every = 50000000
# config.eval_every = 1000

# config.model_type = 'co-attention'
config.reg_lambda = 0.00007
config.names = ["train_context","train_question"]
config.print_every = 100