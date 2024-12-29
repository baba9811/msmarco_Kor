## MS MARCO Korean Dataset
### 한국어 embedding/retrieval test 용 데이터셋
### https://huggingface.co/datasets/namespace-Pt/msmarco
### 데이터셋을 GPT-o mini 로 번역하여 데이터셋 구축하였습니다.

## How to Read?

### 한국어 데이터셋
'''python
with open("trn_msmarco.json") as f:
    data  = json.load(f)
'''

### 영문 데이터셋
인코딩이 깨져있는 단어들이 있어서 json.load만 사용하면
로드가 안됩니다.
'''python
data = []
with open("msmarco.json", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line.strip()))
'''
