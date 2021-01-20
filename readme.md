# Readme

## 설치 방법

### 요구 사항

- apex
- transformers
- datasets
- faiss

저는 NGC 컨테이너 [nvcr.io/nvidia/pytorch:20.12-py3](http://nvcr.io/nvidia/pytorch:20.12-py3) 를 사용해서 conda 기반 환경을 통해 작업을 진행했습니다

### RTX 3090 설치 이슈

faiss 설치를 이와 같이 해야합니다

[https://github.com/facebookresearch/faiss/issues/1524#issuecomment-758688833](https://github.com/facebookresearch/faiss/issues/1524#issuecomment-758688833)

## 파일 구성

### 노트북 순서

1. BERT 구조 및 학습(fine-tune)

    1-1_bert_dev

    1-2_bert_train

2. sparse retrieval(TF-IDF) 및 dense retrieval 구현

    2-1_sparse(tfidf)

    2-2_dense(dpr)

3. sparse와 fine-tuned BERT 를 이용해 시스템 구현 

    3-1_sparse(tfidf)_bert(fine-tuned)

    3-2_dense(dpr)_bert(fine-tuned, rerank)

4. (TO-DO) dense retrieval(DPR) 과 fine-tuned BART decoder를 이용해 시스템 구현 
    - RAG 느낌
5. (TO-DO) phrase retrieval(fine tuned DPR+faiss 으로 예상)을 이용해 시스템 구현
    - DenSPI 같은 느낌

 

### 데이터 구조

```python
./data/        # 전체 데이터
	./dense/         # dense 관련 파일
	./sparse/        # sparse 관련 파일
	./squad/         # squad 샘플 데이터셋
	./squad_odqa/    # squad를 open domain qa 형식으로 바꾼 파일
./output/       # 모델 저장 및 기타. config.py 에서 경로 수정가능 

config.py       # 모델 configuration 저장. 현재는 HF config 와 연동안됨
retrieval.py    # dense/sparse retrieval 모델 구현
utils.py        # 기타 함수 구현
```