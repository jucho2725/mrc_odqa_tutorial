from sklearn.feature_extraction.text import TfidfVectorizer

from scipy import spatial
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from tqdm.notebook import trange, tqdm
import pandas as pd
import pickle
import json
import os
import time

from datasets import Features, Sequence, Value, load_dataset
from typing import Iterable, List, Optional, Tuple
import faiss

import torch
from functools import partial
import numpy as np

from utils import read_file, save_json

# import nltk
# nltk.download('punkt')

from transformers.models.rag.retrieval_rag import CustomHFIndex
from transformers import (
DPRConfig,
DPRQuestionEncoderTokenizer,
DPRQuestionEncoder)

import config as cfg

class SparseRetrieval:
    def __init__(self, mode, data_path=None):
        if not data_path:
            self.data_path = cfg.squad_dir
        else:
            self.data_path = data_path
        self.mode = mode

    def make_embedding(self, context_name):
        # Pickle save.
        pickle_name = "sparse"+ "_" + self.mode + "_embedding.bin"
        pickle_path = os.path.join(cfg.sparse_dir, pickle_name)

        context_dict = read_file(os.path.join(self.data_path, context_name))
        context = context_dict['text']

        tfidfv = TfidfVectorizer(tokenizer=self.tokenize, ngram_range=(1,2)).fit(context)

        if os.path.isfile(pickle_path):
            with open(pickle_path, "rb") as file:
                context_embeddings = pickle.load(file)
            print("Embedding pickle load.")
        else:
            context_embeddings = tfidfv.transform(context).toarray()
            # Pickle save.
            with open(pickle_path, "wb") as file:
                pickle.dump(context_embeddings, file)
            print("Embedding pickle saved.")
        return tfidfv, context_embeddings

    def sparse_searching(self, sparse_query, sparse_embedding, texts, topk=1):
        distances = spatial.distance.cdist(sparse_query, sparse_embedding, "cosine")[0]
        result = zip(range(len(distances)), distances)
        result = sorted(result, key=lambda x: x[1])
        
        cand_dict = {}
        candidate = []
        cand_ids = []
        for idx, distances in result[0:topk]: # top k 
            candidate.append(texts[idx])
            cand_ids.append(idx)

        cand_dict['text'] = candidate
        cand_dict['ids'] = cand_ids
        return cand_dict

    def tokenize(self, text):
        stemmer = PorterStemmer()
        tokens = [word for word in word_tokenize(text)]
        stems = [stemmer.stem(item) for item in tokens]
        return stems

    def retrieve(self, model, sparse_embedding, qas_filename, topk=1):
        # 나중에 비교를 위해 context 정보도 같이 저장
        context_dict = read_file(os.path.join(self.data_path, self.mode + "_context.json"))
        context = context_dict['text']
        context_id = context_dict['ids']

        # make retrieved result as dataframe
        que, que_id, anss, ctxs, ctx_ids = [], [], [], [], []

        # load qas file.
        qas = json.load(open(os.path.join(self.data_path, qas_filename)))['data']
        for item in tqdm(qas):
            query = item['question']
            query_id = item['id']
            answers = item['answers'][0]
            query_s_embedding = model.transform([query]).toarray()
            predict_dict = self.sparse_searching(query_s_embedding,
                                                 sparse_embedding,
                                                 context,
                                                 topk)
            que.append(query)
            que_id.append(query_id)
            anss.append(answers)
            ctxs.append(predict_dict['text'])

            tmp_ctx_ids = [context_id[predict_dict['ids'][i]] for i in range(len(predict_dict['ids']))] # retrieved 된 top k 개 context 들
            ctx_ids.append(tmp_ctx_ids)

        cqas = pd.DataFrame({
            'question': que,
            'que_id': que_id,
            'answers': anss,
            'context': ctxs,          # retrieved documents
            'context_id': ctx_ids,
        })
        return cqas


class DenseRetrieval:
    def __init__(self, config, question_encoder_tokenizer, question_encoder, mode, index=None):
        self.question_encoder_tokenizer = question_encoder_tokenizer
        self.question_encoder = question_encoder

        self.n_docs = config.n_docs
        self.batch_size = config.retrieval_batch_size  # TO DO : batch 과정이 어려울 것 같으면 처음엔 batch 1 로
        self.config = config
        self.index = index

        self.data_path = cfg.squad_dir
        self.mode = mode

    @classmethod
    def from_pretrained(cls, retriever_name_or_path, passages_path, index_path, mode, indexed_dataset=None, **kwargs):
        config = DPRConfig.from_pretrained(retriever_name_or_path, **kwargs)
        # TODO: _build_index 이미 되어있는 경우만 가정 해둠
        assert os.path.isfile(passages_path) or os.path.isfile(index_path)

        # add more attr in config (from rag config)
        """
            retrieval_vector_size (:obj:`int`, `optional`, defaults to 768):
                Dimensionality of the document embeddings indexed by :class:`~transformers.RagRetriever`.
            retrieval_batch_size (:obj:`int`, `optional`, defaults to 8):
                Retrieval batch size, defined as the number of queries issues concurrently to the faiss index encapsulated
            :class:`~transformers.RagRetriever`.
            n_docs (:obj:`int`, `optional`, defaults to 5):
                Number of documents to retrieve.
        """
        config.retrieval_vector_size = 768
        config.retrieval_batch_size = 8
        config.n_docs = 5

        question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            retriever_name_or_path, config=config)
        question_encoder = DPRQuestionEncoder.from_pretrained(retriever_name_or_path)

        if not indexed_dataset:
            print("searching indexed dataset")
            if os.path.isdir(passages_path) and os.path.isfile(index_path):
                print("passages and index is exist")
                from datasets import load_from_disk
                indexed_dataset = load_from_disk(passages_path)  # to reload the dataset
                indexed_dataset.load_faiss_index("embeddings", index_path)  # to reload the index
            else:
                indexed_dataset = cls._build_index(cls, index_path)
        else:
            raise NotImplementedError

        index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
        return cls(
            config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            question_encoder=question_encoder,
            mode=mode,
            index=index,
        )

    def save_pretrained(self, save_directory):
        if isinstance(self.index, CustomHFIndex):
            if self.config.index_path is None:
                index_path = os.path.join(save_directory, "hf_dataset_index.faiss")
                self.index.dataset.get_index("embeddings").save(index_path)
                self.config.index_path = index_path
            if self.config.passages_path is None:
                passages_path = os.path.join(save_directory, "hf_dataset")
                # datasets don't support save_to_disk with indexes right now
                faiss_index = self.index.dataset._indexes.pop("embeddings")
                self.index.dataset.save_to_disk(passages_path)
                self.index.dataset._indexes["embeddings"] = faiss_index
                self.config.passages_path = passages_path
        self.config.save_pretrained(save_directory)
        self.question_encoder_tokenizer.save_pretrained(save_directory)


    def _load_json(self, path):
        with open(f'{path}', 'r') as reader:
            input_data = json.load(reader)['data']
        row_list = []
        count = 1
        for entry in input_data:
            title = entry['title']
            paragraphs = entry['paragraphs']
            for _ in range(len(paragraphs[0])):
                context_text = paragraphs[0][f'context{count}']
                count += 1
                temp = {
                    'title': title,
                    'text': context_text,
                }
                row_list.append(temp)
        df = pd.DataFrame(row_list)
        return df

    def _chunk_tensor(self, t: Iterable, chunk_size: int) -> List[Iterable]:
        return [t[i: i + chunk_size] for i in range(0, len(t), chunk_size)]

    def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, np.ndarray]:
        question_hidden_states_batched = self._chunk_tensor(question_hidden_states,
                                                            self.batch_size)
        ids_batched = []
        vectors_batched = []
        for question_hidden_states in question_hidden_states_batched:
            ids, vectors = self.index.get_top_docs(question_hidden_states, n_docs)
            ids_batched.extend(ids)
            vectors_batched.extend(vectors)
        return (
            np.array(ids_batched),
            np.array(vectors_batched),
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def dense_searching(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:
        """
        Retrieves documents of a query for specified ``question_hidden_states``.

        Args:
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`):
                A batch of query vectors to retrieve with.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Return:
            :obj:`Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:

            - **retrieved_doc_embeds** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs, dim)`) -- The retrieval
              embeddings of the retrieved docs per query.
            - **doc_ids** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`) -- The ids of the documents in the
              index
            - **doc_dicts** (:obj:`List[dict]`): The :obj:`retrieved_doc_embeds` examples per query.
        """

        doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
        return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids)

    def retrieve(self, qas_filename, topk=1):
        # 나중에 비교를 위해 context 정보도 같이 저장
        context_dict = read_file(os.path.join(self.data_path, self.mode + "_context.json"))
        context = context_dict['text']
        context_id = context_dict['ids']

        # make retrieved result as dataframe
        que, que_id, anss, ctxs, ctx_ids = [], [], [], [], []

        # load qas file.
        qas = json.load(open(os.path.join(self.data_path, qas_filename)))['data']
        for item in tqdm(qas):
            query = item['question']
            query_id = item['id']
            answers = item['answers'][0]

            input_ids = self.question_encoder_tokenizer(query, return_tensors="pt")["input_ids"]
            question_hidden_states = self.question_encoder(input_ids)[0]
            retrieved_doc_embeds, doc_ids, doc_dicts = self.dense_searching(
                question_hidden_states.cpu().detach().to(torch.float32).numpy(),
                n_docs=topk)

            que.append(query)
            que_id.append(query_id)
            anss.append(answers)
            ctxs.append(doc_dicts[0]['text'][0])
            # TODO: HFINDEX 의 doc_id 와 저장된 doc_id 가 일치하지 않음
            ctx_ids.append([doc_ids.tolist()[0][i] for i in range(len(doc_ids.tolist()[0]))])

        cqas = pd.DataFrame({
            'question': que,
            'que_id': que_id,
            'answers': anss,
            'context': ctxs,
            'context_id': ctx_ids
        })
        return cqas


    # TODO: _build_index 이미 되어있는 경우만 가정 해둠
    def _build_index(self, path):
        return
    #     df = self._load_json(path)
    #     df.to_csv(path, sep='\t', index=False)
    #     dataset = load_dataset(
    #         "csv", data_files=[f"{output_dir}/train_tc.csv"], split="train", delimiter="\t",
    #         column_names=["title", "text"]
    #     )
    #     dataset = dataset.map(split_documents, batched=True, num_proc=4)
    #     ctx_encoder = DPRContextEncoder.from_pretrained(dpr_ctx_encoder_model_name).to(device=device)
    #     ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(dpr_ctx_encoder_model_name)
    #     new_features = Features(
    #         {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
    #     )  # optional, save as float32 instead of float64 to save space
    #     dataset = dataset.map(
    #         partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
    #         batched=True,
    #         batch_size=batch_size,
    #         features=new_features,
    #     )
    #     passages_path = os.path.join(output_dir, "my_knowledge_dataset")
    #     dataset.save_to_disk(passages_path)
    #     dim = 768  # The dimension of the embeddings to pass to the HNSW Faiss index.
    #     num = 128  # The number of bi-directional links created for every new element during the HNSW index construction.
    #
    #     index = faiss.IndexHNSWFlat(dim, num, faiss.METRIC_INNER_PRODUCT)
    #     dataset.add_faiss_index("embeddings", custom_index=index)
    #     index_path = os.path.join(output_dir, "my_knowledge_dataset_hnsw_index.faiss")
    #     dataset.get_index("embeddings").save(index_path)
    #     return


if __name__ == "__main__":
    # Test sparse
    mode = 'dev'
    context_file = f'{mode}_context.json'
    qa_file =  f'{mode}_qa.json'
    ret_sp = SparseRetrieval(mode, data_path=cfg.squad_dir)
    tfidfv, context_embeddings = ret_sp.make_embedding(context_file)
    cqa_df = ret_sp.retrieve(tfidfv, context_embeddings, qa_file)
    res_path = os.path.join(cfg.sparse_dir, f"retrieved_{mode}_sparse.csv")
    cqa_df.to_csv(res_path, sep='\t', index=False)

    # Test dense
    model_name = 'facebook/dpr-question_encoder-single-nq-base'
    passages_path = os.path.join(cfg.dense_dir, "dpr_dataset")
    index_path = os.path.join(cfg.dense_dir, "dpr_dataset_hnsw_index.faiss")
    ret_ds = DenseRetrieval.from_pretrained(model_name, passages_path=passages_path, index_path=index_path, mode=mode)
    cqa_df = ret_ds.retrieve(qa_file)
    res_path = os.path.join(cfg.dense_dir, f"retrieved_{mode}_dense.csv")
    cqa_df.to_csv(res_path, sep='\t', index=False)
