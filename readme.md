# Readme

## 소개

MRC, open-domainQA 를 구현해놓은 레포입니다.

## 설치 방법

### 요구 사항

- transformers
- datasets
- faiss, apex

transformers 와 datasets 는 공식 installation 을 따르면 됩니다

### apex 관련 
Automatic Mixed Precision(AMP)를 사용하기 위해 apex가 필요합니다

저는 Linux Mint Tricia(Ubuntu18.04 기반) 에서 NGC 컨테이너 [nvcr.io/nvidia/pytorch:20.12-py3](http://nvcr.io/nvidia/pytorch:20.12-py3) 를 사용해서 conda 기반 환경을 통해 작업을 진행했습니다. 

해당 컨테이너는 apex가 설치가 되어있어 문제가 없었으나 혹시 로컬에서 직접 실행하실때 이슈가 있다면 말해주세요 

Colab 사용시 apex 설치

```python
try:
  import apex
except Exception:
  ! git clone https://github.com/NVIDIA/apex.git
  % cd apex
  !pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
  %cd ..
```

### faiss 관련

faiss 는 conda 기반으로 설치하는 것이 가장 편합니다. 
직접 빌드하는 경우 생기는 오류들은 깃헙 이슈를 참고하시면 도움이 됩니다.
cuda 11 은 아직 공식적으로 지원하지 않습니다. 다만 cuda 버전 지정없이 설치시 잘 작동합니다

### RTX 3090 사용시 faiss 설치 이슈

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

3. sparse와 fine-tuned BERT 를 이용해 ODQA 시스템 구현 

    3-1_sparse(tfidf)_bert(fine-tuned)

    3-2_dense(dpr)_bert(fine-tuned, rerank)

4. (TO-DO) dense retrieval(DPR) 과 fine-tuned BART decoder를 이용해 시스템 구현 
    - RAG 느낌
5. (TO-DO) phrase retrieval(fine tuned DPR+faiss 으로 예상)을 이용해 시스템 구현
    - DenSPI 같은 느낌
6. (TO-DO) generative 모델로 closed-book QA 하기 
    - BART, T5, REALM

 

### 저장소 구조

```python
./data/        # 전체 데이터
	./dense/         # dense 관련 파일
	./sparse/        # sparse 관련 파일
	./squad/         # squad 샘플 데이터셋 (train, dev) 및 
                         # squad를 open domain qa 형식으로 바꾼 파일 (_context, _qa)
./output/       # 모델 저장 및 기타. config.py 에서 경로 수정가능 

config.py       # 모델 configuration 저장. 현재는 HF config 와 연동안됨
retrieval.py    # dense/sparse retrieval 모델 구현
utils.py        # 기타 함수 구현
```

## 데이터 소개

train/dev 를 만들기 위해 squad v1.1 데이터중 200/20 개의 title만 샘플링했습니다 

squad 데이터는 title-context-question,answer 으로 나눌수 있습니다 

아래는 train, dev 의 분포를 보여줍니다
```python
# train.json
title length 200      # 총 200 개의 title에서 context를 뽑았고  
context length 8490   # 총 8490 개의 context를 보고 question/answer을 만들었으며
question length 38708 # 결국 38708 개의 학습 데이터(c,q,a)가 만들어짐
```
```python
# dev.json
title length 20       # 같은 방식
context length 920
question length 4639
```

Open-domain QA 형태로 바꾸기위해, 원래 데이터를 다음과 같이 context와 question&answer로 나누었습니다.
```python
# dev_context.json
예시 넣어야함 
```
```python
# dev_qa.json
예시 넣어야함 
```
Open-domain QA 는 현재 데이터에서 다음과 같은 순서 진행됩니다.

1. XXX_qa.json 에서 question 이 query 가 되고, answer은 정답이 됨
2. 이를 기준으로 XXX_context.json 내에서 가장 관련있는 context를 찾고, 이를 (c,q,a) 형태로 묶어서 다시 XXXXXX.json 으로 만들어줌 
3. XXXXX.json 파일 가지고 일반적인 MRC 처럼 수행 후 answer과 비교하여 EM, F1_score 계산

- closed-bookQA 를 위해선 context 를 찾는 과정을 제외하면 됩니다


## TODO

자료 작성 관련 
- RAG DenSPI 추가 
- T5 REALM 추가
- fine-tune 가능하도록 만들기
- HF에서 그냥 불러온 것들 중 이해해야하는 것들은 직접 구현해야함


코드 관련
- reformat, refactor, clean 
- 오래걸리는 부분 multiprocessing 적용 
   - dense/sparse retrieve 부분 
   - squad_evaluate 부분 
   - get_raw_score 문제 해결 
   
기타 
- 데이터 선별 무작위로 다시 
- 데이터 더 크게?




 