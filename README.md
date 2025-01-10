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

  
**평가지표:** HitRate, Recall, MAP(Mean Average Precision), NDCG(Normalized Discounted Cumulative Gain), MRR (Mean Reciprocal Rank), 파라미터 수, Evaluation 소요시간

| **Model** | **HitRate   @1(5)** | **Recall   @1(5)** | **MAP   @1(5)** | **NDCG   @1(5)** | **MRR   @1(5)** | **Time(sec.)** | **\# of   Params** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nlpai-lab/KURE-v1 | **0.936**   **(0.984)** | 0.917   **(0.983)** | 0.917   **(0.955)** | **0.936**  **(0.963)** | **0.936**   **(0.957)** | 11.14 | 568M |
| dragonkue/BGE-m3-ko | **0.936**   **(0.984)** | **0.918**   **(0.983)** | **0.918**   **(0.955)** | **0.936**  **(0.963)** | **0.936**   **(0.957)** | 11.02 | 568M |
| BAAI/bge-m3 | 0.930   **(0.984)** | 0.912   **(0.983)** | 0.912   (0.952) | 0.930  (0.961) | 0.930   (0.954) | 15.59 | 568M |
| jinaai/jina-embeddings-v3 | 0.911   (0.981) | 0.893   (0.979) | 0.893   (0.940) | 0.911   (0.950) | 0.911   (0.941) | 9.12 | 572M |
| upskyy/bge-m3-korean | 0.898   (0.971) | 0.881   (0.970) | 0.881   (0.928) | 0.898   (0.939) | 0.898   (0.930) | 11.08 | 568M |
| intfloat/multilingual-e5-large-instruct | 0.848   (0.963) | 0.830   (0.961) | 0.830   (0.894) | 0.848   (0.912) | 0.848   (0.897) | 11.04 | 558M |
| bespin-global/klue-sroberta-base-continue-learning-by-mnr | 0.701   (0.869) | 0.686   (0.865) | 0.686   (0.765) | 0.701   (0.792) | 0.701   (0.770) | **4.62** | 110M |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 0.544   (0.714) | 0.531   (0.709) | 0.531   (0.606) | 0.544   (0.633) | 0.544   (0.610) | 4.72 | 278M |
| OpenAI/text-embedding-3-small | 0.787   (0.907) | 0.770   (0.903) | 0.770   (0.831) | 0.787   (0.850) | 0.787   (0.835) | \- | \- |
| OpenAI/text-embedding-3-large | 0.877   (0.964) | 0.861   (0.962) | 0.861   (0.912) | 0.877   (0.926) | 0.877   (0.914) | \- | \- |
| Okapi BM25 | 0.509   (0.649) | 0.496   (0.643) | 0.496   (0.557) | 0.509   (0.580) | 0.509   (0.563) | \- | \- |

#### 영문 데이터셋
영문 데이터셋 평가 모델  
**from huggingface:** BAAI/bge-m3, jinaai/jina-embeddings-v3, sentence-transformers/paraphrase-multilingual-mpnet-base-v2, intfloat/multilingual-e5-large-instruct  
  
**openAI:** text-embedding-3-small, text-embedding-3-large  
  
**Keyword based:** Okapi BM25

| **Model** | **HitRate   @1(5)** | **Recall   @1(5)** | **MAP   @1(5)** | **NDCG   @1(5)** | **MRR   @1(5)** | **Time(sec.)** | **\# of   Params** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BAAI/bge-m3 | 0.944   (0.992) | 0.944   (0.992) | 0.944   (0.965) | 0.944   (0.972) | 0.944   (0.965) | 15.68 | 568M |
| jinaai/jina-embeddings-v3 | 0.940   (0.993) | 0.940   (0.993) | 0.940   (0.964) | 0.940   (0.972) | 0.940   (0.964) | 8.89 | 572M |
| intfloat/multilingual-e5-large-instruct | 0.945   (0.992) | 0.945   (0.992) | 0.945   (0.967) | 0.945   (0.973) | 0.945   (0.966) | 9.69 | 560M  |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 0.846   (0.959) | 0.846   (0.959) | 0.846   (0.893) | 0.846   (0.910) | 0.846   (0.893) | 16.78 | 278M |
| OpenAI/text-embedding-3-small | 0.952   (0.996) | 0.952   (0.996) | 0.952   (0.972) | 0.952   (0.978) | 0.952   (0.971) | \- | \- |
| OpenAI/text-embedding-3-large | **0.962**   **(0.997)** | **0.962**   **(0.997)** | **0.962**   **(0.979)** | **0.962**   **(0.984)** | **0.962**   **(0.978)** | \- | \- |
| Okapi BM25 | 0.363   (0.497) | 0.363   (0.497) | 0.363   (0.414) | 0.363   (0.435) | 0.363   (0.414) | \- | \- |