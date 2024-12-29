## MS MARCO Korean Dataset
### 한국어 embedding/retrieval test 용 데이터셋
### https://huggingface.co/datasets/namespace-Pt/msmarco
### 데이터셋을 GPT-o mini 로 번역하여 데이터셋 구축하였습니다.
데이터셋은 id, query, positive로 구성되어있습니다.  
id: 1번부터 6,980번까지 각 query-positive pair의 id  
query: positive를 검색할 query  
positive: query에 대해 검색될 문장/문단으로 구성된 리스트

## How to Read?

```python
# Korean dataset
with open("msmarco_kor.json") as f:
    data  = json.load(f)

# English dataset
with open("msmarco_eng.json") as f:
    data = json.load(f)
```

### Retrieval Evaluation
huggingface에서 불러올 수 있는 모델들이랑, GPT embedding model을 비교해봤습니다.  
벡터 인덱싱에는 faiss 모델을 사용했습니다.

#### 한글 데이터셋
한글 임베딩 데이터셋 평가 모델  

**from huggingface:** BAAI/bge-m3, upskyy/bge-m3-korean, dragonkue/BGE-m3-ko, jinaai/jina-embeddings-v3, bespin-global/klue-sroberta-base-continue-learning-by-mnr, intfloat/multilingual-e5-large-instruct, sentence-transformers/paraphrase-multilingual-mpnet-base-v2  
  
**openAI:** text-embedding-3-small, text-embedding-3-large  
  
**Keyword based:** Okapi BM25

  
**평가지표:** 데이터 임베딩 시간 (약 7,000개), 파라미터 수, HitRate, Recall, MAP(Mean Average Precision), NDCG(Normalized Discounted Cumulative Gain), MRR (Mean Reciprocal Rank)

환경은 모두 Colab A100 GPU, 데이터 임베딩 batch_size는 32로 설정하고 실험했습니다.

| **Model** | **HitRate   @1(5)** | **Recall   @1(5)** | **MAP   @1(5)** | **NDCG   @1(5)** | **MRR   @1(5)** | **Time(sec.)** | **\# of   Params** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| intfloat/multilingual-e5-large-instruct | 0.795   (0.942) | 0.795   (0.942) | 0.795   (0.891) | 0.795   (0.900) | 0.795   (0.856) | 60.95 | 560M |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 0.533   (0.707) | 0.533   (0.707) | 0.533   (0.622) | 0.533   (0.641) | 0.533   (0.601) | 19.24 | 278M |
| bespin-global/klue-sroberta-base-continue-learning-by-mnr | 0.711   (0.873) | 0.711   (0.873) | 0.711   (0.873) | 0.711   (0.819) | 0.711   (0.777) | 18.58 | 110M |
| jinaai/jina-embeddings-v3 | 0.910   **(0.979)** | 0.910   **(0.979)** | 0.910   **(0.977)** | 0.910   **(0.973)** | 0.910   **(0.940)** | **9.99** | 572M |
| BAAI/bge-m3 | **0.912   **(0.976) | **0.912   **(0.976) | **0.912   (0.977)** | **0.912   **(0.972) | **0.912   **(0.939) | 60.97 | 568M |
| upskyy/bge-m3-korean | 0.898   (0.971) | 0.898   (0.971) | 0.898   (0.966) | 0.898   (0.963) | 0.898   (0.930) | 61.02 | 568M |
| dragonkue/BGE-m3-ko | 0.908   (0.975) | 0.908   (0.975) | 0.908   (0.973) | 0.908   (0.969) | 0.908   (0.936) | 61.00 | 568M |
| OpenAI/text-embedding-3-small | 0.785   (0.907) | 0.785   (0.907) | 0.785   (0.865) | 0.785   (0.872) | 0.785   (0.835) | \- | \- |
| OpenAI/ text-embedding-3-large | 0.877   (0.964) | 0.877   (0.964) | 0.877   (0.950) | 0.877   (0.949) | 0.877   (0.914) | \- | \- |
| Okapi BM25 | 0.509   (0.649) | 0.509   (0.649) | 0.509   (0.581) | 0.509   (0.597) | 0.509   (0.563) | \- | \- |

#### 영문 데이터셋
영문 데이터셋 평가 모델  
**from huggingface:** BAAI/bge-m3, jinaai/jina-embeddings-v3, sentence-transformers/paraphrase-multilingual-mpnet-base-v2, intfloat/multilingual-e5-large-instruct  
  
**openAI:** text-embedding-3-small, text-embedding-3-large  
  
**Keyword based:** Okapi BM25

| **Model** | **HitRate   @1(5)** | **Recall   @1(5)** | **MAP   @1(5)** | **NDCG   @1(5)** | **MRR   @1(5)** | **Time(sec.)** | **\# of   Params** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAAI/bge-m3 | **0.767   (0.903)** | **0.767   (0.903)** | **0.767   (0.865)** | **0.767   (0.870)** | **0.767   (0.822)** | 52.50 | 568M |
| jinaai/jina-embeddings-v3 | 0.752   (0.887) | 0.752   (0.887) | 0.752   (0.850) | 0.752   (0.855) | 0.752   (0.807) | **10.02** | 572M |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 0.492   (0.679) | 0.492   (0.679) | 0.492   (0.595) | 0.492   (0.613) | 0.492   (0.565) | 16.78 | 278M |
| intfloat/multilingual-e5-large-instruct | 0.745   (0.892) | 0.745   (0.892) | 0.745   (0.848) | 0.745   (0.855) | 0.745   (0.805) | 53.02 | 560M  |
| OpenAI/text-embedding-3-small | 0.294   (0.442) | 0.294   (0.442) | 0.294   (0.365) | 0.294   (0.383) | 0.294   (0.350) | \- | \- |
| OpenAI/ text-embedding-3-large | 0.715   (0.866) | 0.715   (0.866) | 0.715   (0.817) | 0.715   (0.825) | 0.715   (0.775) | \- | \- |
| Okapi BM25 | 0.033   (0.044) | 0.033   (0.044) | 0.033   (0.039) | 0.033   (0.040) | 0.033   (0.037) | \- | \- |