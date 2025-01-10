## MS MARCO Korean Dataset
### 한국어 embedding/retrieval test 용 데이터셋
#### https://huggingface.co/datasets/namespace-Pt/msmarco
#### 데이터셋을 GPT-4o mini 로 번역하여 데이터셋 구축하였습니다.
번역에 사용된 프롬프트는 아래와 같습니다.
```
{"role":"system",
"content": "You are a helpful assistant.\
You will get query and chunk pair. You have to translate both into natural Korean.\
Ensure that the output sentence does not contain the escape string.\
Escape strings should be converted in to the corresponding string.\
If there are broken encoding string, be sure to restore and print.\
And the ouput should be in the form of {\"Query\": [translated query],\"Chunk\": [translated chunk]}"}
```

데이터셋은 id, query, positive로 구성되어있습니다.  
id: 1번부터 6,980번까지 각 query-positive pair의 id  
query: positive를 검색할 query  
positive: query에 대해 검색될 문장/문단으로 구성된 리스트

## How to Load?

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
환경은 모두 RTX 3090 24GB GPU, 데이터 임베딩 batch_size는 8로 설정하고 실험했습니다.  
작은 GPU로 테스트하다 보니 용량이 큰 임베딩 모델은 사용을 하지 못해서 MTEB 상위권 모델들은 다수 빠져있습니다.


#### 한글 데이터셋
한글 임베딩 데이터셋 평가 모델  

**from huggingface:** nlpai-lab/KURE-v1, BAAI/bge-m3, upskyy/bge-m3-korean, dragonkue/BGE-m3-ko, jinaai/jina-embeddings-v3, bespin-global/klue-sroberta-base-continue-learning-by-mnr, intfloat/multilingual-e5-large-instruct, sentence-transformers/paraphrase-multilingual-mpnet-base-v2  
  
**openAI:** text-embedding-3-small, text-embedding-3-large  
  
**Keyword based:** Okapi BM25

  
**평가지표:** 데이터 임베딩 시간 (약 7,000개), 파라미터 수, HitRate, Recall, MAP(Mean Average Precision), NDCG(Normalized Discounted Cumulative Gain), MRR (Mean Reciprocal Rank)

| **Model** | **HitRate   @1(5)** | **Recall   @1(5)** | **MAP   @1(5)** | **NDCG   @1(5)** | **MRR   @1(5)** | **Time(sec.)** | **\# of   Params** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nlpai-lab/KURE-v1 | **0.936**   **(0.984)** | **0.936**   **(0.984)** | **0.936**   **(0.995)** | **0.936**  **(0.987)** | **0.936**   **(0.957)** | 11.00 | 568M |
| dragonkue/BGE-m3-ko | **0.936**   **(0.984)** | **0.936**   **(0.984)** | **0.936**   **(0.995)** | **0.936**  **(0.987)** | **0.936**   **(0.957)** | 11.02 | 568M |
| BAAI/bge-m3 | 0.930   (0.981) | 0.930**   (0.981) | 0.930   (0.992) | 0.930  (0.986) | 0.930   (0.954) | 15.59 | 568M |
| jinaai/jina-embeddings-v3 | 0.911   (0.981) | 0.911   (0.981) | 0.911   (0.979) | 0.911   (0.975) | 0.911   (0.941) | 9.12 | 572M |
| intfloat/multilingual-e5-large-instruct | 0.848   (0.963) | 0.848   (0.963) | 0.848   (0.932) | 0.848   (0.936) | 0.848   (0.897) | 11.04 | 558M |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 0.544   (0.714) | 0.544   (0.714) | 0.544   (0.632) | 0.544   (0.650) | 0.544   (0.610) | 4.72 | 278M |
| bespin-global/klue-sroberta-base-continue-learning-by-mnr | 0.701   (0.869) | 0.701   (0.869) | 0.701   (0.797) | 0.701   (0.812) | 0.701   (0.770) | 4.62 | 110M |
| upskyy/bge-m3-korean | 0.898   (0.971) | 0.898   (0.971) | 0.898   (0.966) | 0.898   (0.963) | 0.898   (0.930) | 11.08 | 568M |
| OpenAI/text-embedding-3-small | 0.785   (0.907) | 0.785   (0.907) | 0.785   (0.865) | 0.785   (0.872) | 0.785   (0.835) | \- | \- |
| OpenAI/text-embedding-3-large | 0.877   (0.964) | 0.877   (0.964) | 0.877   (0.950) | 0.877   (0.949) | 0.877   (0.914) | \- | \- |
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
| OpenAI/text-embedding-3-large | 0.715   (0.866) | 0.715   (0.866) | 0.715   (0.817) | 0.715   (0.825) | 0.715   (0.775) | \- | \- |
| Okapi BM25 | 0.033   (0.044) | 0.033   (0.044) | 0.033   (0.039) | 0.033   (0.040) | 0.033   (0.037) | \- | \- |