# coding=utf-8
# @Author       : Jin Uk, Cho
# @Project      : MRC_prep
# @FileName     : config.py
# @Time         : Created at 2021-01-14
# @Description  :
# Copyrights (C) 2021. All Rights Reserved.

import os
import torch

# ===file IO===
data_dir = "./data"
squad_dir = "./data/squad"
dense_dir = "./data/dense"
sparse_dir = "./data/sparse"
output_dir = "./output"
dev_file = "dev.json"
train_file = "train.json"

# test_file = 

# ===model config===
model_name = "bert-base-cased"
tokenizer_name = "bert-base-cased"

doc_stride = 128                # When splitting up a long document into chunks, how much stride to take between chunks.
max_query_length = 64           # The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
max_seq_length = 384            # The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.
n_best_size = 20                # The total number of n-best predictions to generate in the nbest_predictions.json output file.
verbose_logging = False         # If true, all of the warnings related to data processing will be printed.
null_score_diff_threshold = 0.0 # "If null_score - best_non_null is greater than the threshold predict null.

# ===eval config===
eval_batch_size = 4
max_answer_length = 30          # The maximum length of an answer that can be generated. This is needed because the start

# ===train config===
train_batch_size = 4            # Batch size per GPU/CPU for training.
learning_rate = 5e-5            # The initial learning rate for optimizer
num_train_epochs = 5.0          # Total number of training epochs to perform.

gradient_accumulation_steps = 2 # Number of updates steps to accumulate before performing a backward/update pass.
weight_decay = 0.0              # Weight decay if we apply some.
epsilon = 1e-8                  # Epsilon for optimizer.
max_grad_norm = 1.0             # Max gradient norm.


# ===HW config===
threads= 4                      # multiple threads for converting example to features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===Reproducibility ===
seed = 1001

def init_param(args):
    """
    TO DO: initialize parameters from argument objects. will use this for script execution (not ipynb).
    """
    
    global data_dir
    data_dir = args.data_dir
