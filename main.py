import tensorflow as tf
import torch

import numpy as np
import random
import time
import datetime

from model import dataloader, module


if __name__ == "__main__":
    # Configuration Parameters
    train_path = "CleanCode_Bert/myCleanCode/nsmc/ratings_train.txt"
    test_path = "CleanCode_Bert/myCleanCode/nsmc/ratings_test.txt"
    doc_column = 'document'
    label_column = 'label'
    model_name = 'multilingual-cased'
    # 배치 사이즈
    batch_size = 32
    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128

    # default value
    seed_val = 1234

    # 에폭수
    epochs = 4

    # for inference
    # model, config, vocab
    model_path = './model/pt_nsmc_bert_module.pth'

    # def device_settings():
    # GPU 디바이스 이름 구함
    device_name = tf.test.gpu_device_name()

    # GPU 디바이스 이름 검사
    if device_name == '/device:GPU:0':
        print('Found GPU at: {}'.format(device_name))
    else:
        raise SystemError('GPU device not found')

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')

    # 재현을 위해 랜덤시드 고정
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # get data (X, Y) for train, test
    train_sentences, train_labels = dataloader.get_data(train_path, doc_column, label_column)
    test_sentences, test_labels = dataloader.get_data(test_path, doc_column, label_column)

    # load tokenizer, model
    tokenizer, model = module.load_tokenizer_model(model_name)

    # get tokenized train data
    train_tokenized_texts, train_input_ids, train_attention_masks = module.get_tokens_ids(tokenizer, train_sentences)
    # check_tokens(train_tokenized_texts)

    # get train / validation : inputs, labels, masks
    train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = dataloader.split_training_data(
        train_input_ids, train_attention_masks, train_labels)

    # get torch data
    train_dataloader, validation_dataloader = dataloader.make_torch_train_valid_data(train_inputs, validation_inputs, train_labels,
                                                                          validation_labels, train_masks,
                                                                          validation_masks)

    # training model
    model = module.fine_tuning_model(model, train_dataloader, validation_dataloader, model_path)

    # inference, https://tutorials.pytorch.kr/beginner/saving_loading_models.html
    # https://huggingface.co/transformers/v1.2.0/serialization.html
    # 모델 클래스는 어딘가에 반드시 선언되어 있어야 합니다
    model = torch.load(model_path)
    model.eval()

    # get tokenized test data
    test_tokenized_texts, test_input_ids, test_attention_masks = module.get_tokens_ids(tokenizer, test_sentences)
    # 데이터를 파이토치의 텐서로 변환
    test_inputs, test_labels, test_masks = module.make_torch_test_data(test_input_ids, test_labels, test_attention_masks)
    test_dataloader = module.make_test_dataloader(test_inputs, test_labels, test_masks)

    # inference
    module.get_test_result(test_dataloader, model)
    module.get_sample_test_result(model, tokenizer)
    print("test")


