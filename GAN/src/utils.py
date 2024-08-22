import random

import torch
import time
import numpy as np
import pandas as pd
from torch import nn
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from scipy.io import arff
from sklearn.preprocessing import minmax_scale, LabelEncoder
from datetime import datetime

import src


def init_weights(layer: nn.Module):
    layer_name = layer.__class__.__name__
    if 'Linear' in layer_name:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
    elif layer_name == 'BatchNorm1d':
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


def set_random_state(seed: int = None) -> None:
    if seed is None:
        seed = src.config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocess_data(file_name):
    # 파일 경로 설정
    file_path = src.config.path.datasets / file_name
    df = pd.read_csv(file_path)

    # 무작위 샘플링
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Name 열 제거
    df.drop(['Name'], axis=1, inplace=True)

    # 범주형 데이터 레이블 인코딩
    categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Doctor', 'Hospital', 'Insurance Provider', 'Admission Type', 'Medication']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    # 날짜 데이터 처리
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
    df['Admission Duration'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

    # 날짜 열을 다른 형태로 변환
    df['Date of Admission'] = df['Date of Admission'].map(datetime.toordinal)
    df['Discharge Date'] = df['Discharge Date'].map(datetime.toordinal)

    # 샘플과 레이블 분리
    samples = df.drop(columns=['Test Results'])
    labels = df['Test Results']

    # 수치형 데이터 정규화 (min-max 스케일링)
    numerical_cols = ['Age', 'Billing Amount', 'Room Number', 'Admission Duration', 'Date of Admission', 'Discharge Date']
    samples[numerical_cols] = minmax_scale(samples[numerical_cols].astype('float32'))

    # 레이블을 수치형으로 변환
    labels = LabelEncoder().fit_transform(labels)

    # x_size 설정
    src.models.x_size = samples.shape[1]

    return np.array(samples), np.array(labels)

def prepare_dataset(name, training_test_ratio: float = 0.6) -> None:
    samples, labels = preprocess_data(name)
    training_samples, test_samples, training_labels, test_labels = train_test_split(
        samples,
        labels,
        train_size=training_test_ratio,
        random_state=src.config.seed,
    )
    src.datasets.training_samples = training_samples
    src.datasets.training_labels = training_labels
    src.datasets.test_samples = test_samples
    src.datasets.test_labels = test_labels

def get_final_test_metrics(statistics: dict) -> dict:
    metrics = dict()
    for name, values in statistics.items():
        if name == 'Loss':
            continue
        else:
            metrics[name] = values[-1]
    return metrics


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min())


def get_knn_indices(sample: torch.Tensor, all_samples: torch.Tensor, k: int = 5) -> torch.Tensor:
    dist = torch.empty(len(all_samples))
    for i, v in enumerate(all_samples):
        dist[i] = torch.norm(sample - v, p=2)
    return torch.topk(dist, k, largest=False).indices


def get_gan_dataset(gan: src.gans.GANLike) -> src.datasets.FullDataset:

    print("start ganfit")
    gan.fit()
    print("done ganfit")
    
    full_dataset = src.datasets.FullDataset().to(src.config.device)
    pos_dataset = src.datasets.PositiveDataset().to(src.config.device)
    neg_dataset = src.datasets.NegativeDataset().to(src.config.device)
    target_dataset = src.datasets.FullDataset().to(src.config.device)
    # generate positive samples until reaching balance
    total_pos_cnt = len(pos_dataset)
    total_neg_cnt = len(neg_dataset)

    target_sample_num = total_neg_cnt - total_pos_cnt
    if target_sample_num <= 0:
        return full_dataset
    z = torch.rand(target_sample_num, src.models.z_size, device=src.config.device)
    new_samples = gan.generate_samples(z)
    new_labels = torch.ones(target_sample_num, device=src.config.device)
    target_dataset.samples = torch.cat(
        [
            target_dataset.samples,
            new_samples,
        ],
    )
    target_dataset.labels = torch.cat(
        [
            target_dataset.labels,
            new_labels,
        ]
    )
    target_dataset.samples = target_dataset.samples.detach()
    target_dataset.labels = target_dataset.labels.detach()
    return target_dataset


def get_rgan_dataset(rgan: src.gans.GANLike) -> src.datasets.FullDataset:
    vae = src.vae.VAE()
    vae.fit()
    rgan.fit()

    full_dataset = src.datasets.FullDataset().to(src.config.device)
    pos_dataset = src.datasets.PositiveDataset().to(src.config.device)
    neg_dataset = src.datasets.NegativeDataset().to(src.config.device)

    # count negative samples in the overlapping area
    ol_neg_cnt = 0
    iter_count = 0
    for i in neg_dataset.samples:
        print("count neg samples in overlapping area : {}/{}".format(iter_count, len(neg_dataset.samples)))
        indices = get_knn_indices(i, full_dataset.samples)
        labels = full_dataset.labels[indices]
        if 1 in labels:
            ol_neg_cnt += 1
        iter_count += 1

    # count positive samples in the overlapping area
    ol_pos_cnt = 0
    iter_count = 0
    for i in pos_dataset.samples:
        print("count pos samples in overlapping area : {}/{}".format(iter_count, len(neg_dataset.samples)))
        indices = get_knn_indices(i, full_dataset.samples)
        labels = full_dataset.labels[indices]
        if 0 in labels:
            ol_pos_cnt += 1
        iter_count += 1

    target_dataset = src.datasets.FullDataset().to(src.config.device)
    # generate positive samples until reaching balance
    total_pos_cnt = len(pos_dataset)
    total_neg_cnt = len(neg_dataset)

    while True:
        start_time = time.time()
        if total_pos_cnt >= total_neg_cnt or ol_pos_cnt >= ol_neg_cnt:
            break
        else:
            # update the number of positive samples
            z = vae.generate_z()
            new_sample = rgan.generate_samples(z)
            new_label = torch.tensor([1], device=src.config.device)
            target_dataset.samples = torch.cat(
                [
                    target_dataset.samples,
                    new_sample,
                ],
            )
            target_dataset.labels = torch.cat(
                [
                    target_dataset.labels,
                    new_label,
                ]
            )
            total_pos_cnt += 1
            # update the number of overlapping positive samples
            indices = get_knn_indices(new_sample, full_dataset.samples)
            labels = full_dataset.labels[indices]
            if 0 in labels:
                ol_pos_cnt += 1
        
        end_time = time.time()

        print("update the number of positive samples")
        print("total_post_cnt : {}".format(total_pos_cnt))
        print("total_neg_cnt : {}".format(total_neg_cnt))
        print("ol_pos_cnt : {}".format(ol_pos_cnt))
        print("ol_neg_cnt : {}".format(ol_neg_cnt))
        print("TIME : {}".format(end_time-start_time))

    target_dataset.samples = target_dataset.samples.detach()
    target_dataset.labels = target_dataset.labels.detach()

    return target_dataset


def get_jgan_dataset(rgan: src.gans.GANLike) -> src.datasets.FullDataset:
    vae = src.vae.VAE()
    vae.fit()
    rgan.fit()

    pos_dataset = src.datasets.PositiveDataset().to(src.config.device)
    neg_dataset = src.datasets.NegativeDataset().to(src.config.device)
    target_dataset = src.datasets.FullDataset().to(src.config.device)

    # generate positive samples until reaching balance
    total_pos_cnt = len(pos_dataset)
    total_neg_cnt = len(neg_dataset)

    while True:
        start_time = time.time()
        if total_pos_cnt >= total_neg_cnt :
            break
        else:
            # update the number of positive samples
            z = vae.generate_z()
            new_sample = rgan.generate_samples(z)
            new_label = torch.tensor([1], device=src.config.device)
            target_dataset.samples = torch.cat(
                [
                    target_dataset.samples,
                    new_sample,
                ],
            )
            target_dataset.labels = torch.cat(
                [
                    target_dataset.labels,
                    new_label,
                ]
            )
            total_pos_cnt += 1
        
        end_time = time.time()

        print("update the number of positive samples")
        print("total_post_cnt : {}".format(total_pos_cnt))
        print("total_neg_cnt : {}".format(total_neg_cnt))
        print("TIME : {}".format(end_time-start_time))

    target_dataset.samples = target_dataset.samples.detach()
    target_dataset.labels = target_dataset.labels.detach()

    return target_dataset
