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

## TODO

자료 작성 관련 
- RAG DenSPI 추가 
- fine-tune 가능하도록 만들기
- HF에서 그냥 불러온 것들 중 이해해야하는 것들은 직접 구현해야함


코드 관련 
- black
- 리팩토링 
- 오래걸리는 부분 multiprocessing 적용 
   - dense/sparse retrieve 부분
   
기타 
- 데이터 선별 무작위로 다시 
- 데이터 더 크게?




 