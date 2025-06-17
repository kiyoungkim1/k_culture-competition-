# 과제
https://kli.korean.go.kr/benchmark/taskOrdtm/taskDownload.do?taskOrdtmId=181&clCd=ING_TASK&subMenuId=sub02&resultCd=58058&resultCK=SUCCESS#

# 학습
ssh -p 8501 raynor@1.249.212.198 

# 도커 접속
docker exec -it k_culture /bin/bash

# 실행
(환경 테스트) python -m run.main  --input resource/QA/sample_qa.json  --output result.json   --model_id naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B   --device cuda:0
python -m run.main  --input resource/data_given/korean_culture_qa_V1.0_test.json  --output result.json   --model_id naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B   --device cuda:0


# deepseek-ai/DeepSeek-R1-Distill-Qwen-32B   --> 4bit로 불러오면 24GB 안에 돌릴 수 있음

# 평가 데이터 양식
https://github.com/teddysum/korean_evaluation




============================================

# 한국문화 질의응답 Baseline
본 리포지토리는 '2025년 국립국어원 인공지능의 한국어 능력 평가' 경진 대회 과제 중 '한국문화 질의응답'에 대한 베이스라인 모델의 추론과 평가를 재현하기 위한 코드를 포함하고 있습니다.


추론 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.

### Baseline
|           Model           | Accuracy | Exact Match | ROUGE-1 | BERTScore | BLEURT | Descriptive Avg | Final Score |
| :-----------------------: | :------: | :---------: | :-----: | :-------: | :----: | :-------------: | :---------: |
|        **Qwen3-8B**        |  0.4715  |    0.4216   |  0.4109 |   0.7198  | 0.5451 |      0.5586     |    0.4839   |
| **HyperCLOVAX Text 1.5B** |  0.4922  |    0.2941   |  0.3076 |   0.7057  | 0.5325 |      0.5153     |    0.4339   |


 - 선다형: Accuracy
 - 단답형: EM
 - 서술형: ROUGE, BERTScore, BLEURT

   
평가 코드 : https://github.com/teddysum/korean_evaluation.git


## Directory Structure
```
# 평가에 필요한 데이터가 들어있습니다.
resource
└── QA

# 실행 가능한 python 스크립트가 들어있습니다.
run
└── test.py

# 학습에 사용될 커스텀 함수들이 구현되어 있습니다.
src
└── data.py   
```

## Data Format
```
{
    "id": "1",
    "input": {
        "category": "문화 지식",
        "domain": "예술",
        "question_type": "선다형",
        "topic_keyword": "전통놀이",
        "question": "다음 한국의 전통 놀이 중 '조선시대'에 행한 놀이는? \\n 1\\t주사위 놀이 2\\t검무 3\\t격구 4\\t영고 5\\t무애무"
    },
    "output": {
        "answer": "3"
    }
},
```

## How to Run
### Inference
```
python -m run.test \
    --input resource/QA/sample.json \
    --output result.json \
    --model_id Bllossom/llama-3.2-Korean-Bllossom-3B \
    --device cuda:0 \
```



## Reference
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
transformers (https://github.com/huggingface/transformers)  
Bllossome (Teddysum) (https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B)
Qwen3-8B (https://huggingface.co/Qwen/Qwen3-8B)
HyperCLOVAX-SEED-Text-Instruct-1.5B (https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)


