# coding=utf-8

import json
import random
import torch
import numpy as np
import config as cfg
from transformers.data.processors.squad import SquadProcessor, squad_convert_examples_to_features
from typing import Iterable, List, Optional, Tuple
import os

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if (not args.no_cuda) and (torch.cuda.is_available()):
        torch.cuda.manual_seed_all(args.seed)


def read_file(data_path):
    context = json.load(open(data_path))
    cont_dict = {}
    text = []
    ids = []
    for data in context['data']:
        paragraphs = data['paragraphs']
        for para in paragraphs:
            temp = para.values()
            idx = list(para.keys())
            text.extend(temp)
            ids.extend(idx)
    cont_dict['ids'] = ids
    cont_dict['text'] = text
    return cont_dict


def convert_json(df):
    rebuild = {}
    datas = []

    for i in range(len(df)):
        para_dict = {}
        para_dict['context'] = df.loc[i, "context"]

        qas_dict = {}

        qas_dict['question'] = df.loc[i, "question"]
        qas_dict['answers'] = df.loc[i, 'answers']  # should comment out for test
        qas_dict['id'] = df.loc[i, "que_id"]

        para_dict['qas'] = [qas_dict]
        datas.append(para_dict)

    rebuild['data'] = []
    rebuild['data'].append({'title':"converted",
                           'paragraphs': datas})

    return rebuild


def save_json(df, mode_or_filename):
    result_dict = convert_json(df)

    if mode_or_filename == "train" or mode_or_filename == "dev" or mode_or_filename == "test":
        file_path = os.path.join(cfg.squad_dir, f"{mode_or_filename}.json")
    else:
        if 'sparse' in mode_or_filename:
            file_path = os.path.join(cfg.sparse_dir, mode_or_filename)
        elif 'dense' in mode_or_filename:
            file_path = os.path.join(cfg.dense_dir, mode_or_filename)
        print(f"filename: {file_path}" )
    with open(file_path, "w") as json_file:
        json.dump(result_dict, json_file)

class SquadV1Processor(SquadProcessor):
    train_file = cfg.train_file
    dev_file = cfg.dev_file

def load_and_cache_examples(args, tokenizer, mode_or_filename, output_examples=False):
    """
    Changes 
        1. no distributed training(removed for simplicity)
        2. no caching(cache make preprocessing time shorter, but removed for simplicity)
    """
    input_dir = args.squad_dir if args.squad_dir else args.data_dir

    print("Creating features from dataset file at %s", input_dir)

    if mode_or_filename == "train" or mode_or_filename == "dev" or mode_or_filename == "test":
        mode = mode_or_filename
        processor = SquadV1Processor()
        if mode == 'test':
            examples = processor.get_dev_examples(args.squad_dir, filename=processor.test_file)
        elif mode == 'dev':
            examples = processor.get_dev_examples(args.squad_dir, filename=processor.dev_file)
        else:
            examples = processor.get_train_examples(args.squad_dir, filename=processor.train_file)
    else:
        mode = 'dev'
        processor = SquadV1Processor()
        processor.edited_file = mode_or_filename

        if 'sparse' in mode_or_filename:
            examples = processor.get_dev_examples(args.sparse_dir, filename=processor.edited_file)
        elif 'dense' in mode_or_filename:
            examples = processor.get_dev_examples(args.dense_dir, filename=processor.edited_file)
        else:
            print("*" * 50)
            print(processor.edited_file)
            print("*" * 50)
            raise FileNotFoundError

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=True if mode == 'train' else False,
        return_dataset='pt',
        threads=args.threads,
    )

#     torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset

def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]

def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}