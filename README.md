# Clean-Code-python
practice Clean Code In Python

# NLP

# 0. 들어가며

- NLP와 관련한 Task를 클린코드로 구현하는 것이 목적임
- Hugging face 계열의 transfomer 모델 활용하기
- 추가 제안은 코멘트

# 1. Task (TBD)

- 네이버 영화 리뷰 데이터를 활용한 감성분석 (Classification)
- Koquard 2.0 데이터셋을 활용한 NLP
    - 단, 모든 Task를 Try하기엔 규모가 너무 크므로 1~2개 Task로 국한
- 추가 제안은 코멘트

# 2. Transfomer 계열 모델 구조

1. Data Load
    - Raw data load
    - Data Split (train, validation, test)
2. Preprocessing
    - 텍스트 전처리 (정규 표현식, 띄어쓰기, 오타 등)
    - Transfomer 활용을 위한 전처리 (SEP, EOS token 등 삽입)
    - Padding 등 전처리
3. Tokenizing
    - SentencePiece 계열 활용
    - Mecab등 한글에 특화된 Tokenizer 활용
    - 이외 선택사항 다수
    - 단, 일반적으로 Transfomer 계열 사용하는 경우에 Hugginface 모델에 맞는 Tokenizer 사용
4. Task에 맞는 모델 정의
    - Bert, Albert, roberta Load
    - Task에 맞게 모델 수정 & 보완
5. 학습
    - Training
    - Evaluation

# 3. Reference

bert model architecture (with huggingface)

[http://mccormickml.com/2019/07/22/BERT-fine-tuning/](http://mccormickml.com/2019/07/22/BERT-fine-tuning/)

bert 구현 (with pytorch)

[https://paul-hyun.github.io/bert-02/](https://paul-hyun.github.io/bert-02/)

kaggle notebook (with huggingface)

[https://www.kaggle.com/search?q=hugging+face+in%3Anotebooks](https://www.kaggle.com/search?q=hugging+face+in%3Anotebooks)

네이버 영화 리뷰 SA 분석 (with huggingface)

[https://colab.research.google.com/drive/1tIf0Ugdqg4qT7gcxia3tL7und64Rv1dP](https://colab.research.google.com/drive/1tIf0Ugdqg4qT7gcxia3tL7und64Rv1dP)

[https://github.com/monologg/KoBERT-nsmc](https://github.com/monologg/KoBERT-nsmc)
