import tensorflow as tf
import torch

import numpy as np
import random
import argparse
import time
import datetime
import os

from model import dataloader, module

# def device_settings():
# GPU 디바이스 이름 구함
# device_name = tf.test.gpu_device_name()
device_name = torch.cuda.get_device_name()

# GPU 디바이스 이름 검사
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """  """


if __name__ == "__main__":
    # # Configuration Parameters

    # # 입력 토큰의 최대 시퀀스 길이
    # MAX_LEN = 128

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help='Where to store logs and models')
    parser.add_argument('--train_data',required=True, help='path to training dataset')
    parser.add_argument('--test_data',required=True, help='path to test dataset')
    parser.add_argument('--manualSeed',type=int, defalut=1234, help='for random seed setting')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--epoch', type=int, default=4, help='input epoch num')
    parser.add_argument('--saved_model', default='', help='path to model to continue training')

    opt = parser.parse_args()

    # model_path = './model/pt_nsmc_bert_module.pth'

    if not opt.experiment_name:
        model_name = opt.saved_model.split('/')[-1].split('.')[0]
        opt.experiment_name = f'{model_name}'
        opt.experiment_name += f'-Seed{opt.manualSeed}'

    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)

    """ Seed and GPU setting """
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    train(opt)



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


