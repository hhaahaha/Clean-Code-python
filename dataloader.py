import torch
import pandas as pd
from model.dataset import MovieDataSet
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split


def get_data(data_path):
    """ 데이터 로더 """
    train = pd.read_csv(data_path, sep='\t')

    # 리뷰 문장 추출
    sentences = train['document']

    # 라벨 추출
    labels = train['label'].values

    # BERT의 입력 형식에 맞게 변환
    sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]

    return sentences, labels


# split_training_data 와 make_torch_train_valid_data 를 묶을 수 있을 것 같다
def split_training_data(input_ids, attention_masks, labels, seed_val=1234):
    # 훈련셋과 검증셋으로 분리
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                        labels,
                                                                                        random_state=seed_val,
                                                                                        test_size=0.1)

    # 어텐션 마스크를 훈련셋과 검증셋으로 분리
    train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                           input_ids,
                                                           random_state=seed_val,
                                                           test_size=0.1)

    return train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks




def make_torch_train_valid_data(train_inputs, validation_inputs, train_labels, validation_labels, train_masks,
                                validation_masks, batch_size=32):
    # 데이터를 파이토치의 텐서로 변환
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    print(train_inputs[0])
    print(train_labels[0])
    print(train_masks[0])
    print(validation_inputs[0])
    print(validation_labels[0])
    print(validation_masks[0])

    # 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
    # 학습시 배치 사이즈 만큼 데이터를 가져옴
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader


def make_torch_test_data(input_ids, labels, attention_masks):
    # 데이터를 파이토치의 텐서로 변환
    test_inputs = torch.tensor(input_ids)
    test_labels = torch.tensor(labels)
    test_masks = torch.tensor(attention_masks)

    print(test_inputs[0])
    print(test_labels[0])
    print(test_masks[0])

    return test_inputs, test_labels, test_masks


def make_test_dataloader(test_inputs, test_labels, test_masks, batch_size=32):
    # 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
    # 학습시 배치 사이즈 만큼 데이터를 가져옴
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return test_dataloader