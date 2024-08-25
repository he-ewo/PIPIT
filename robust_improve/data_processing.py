import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_data():
    # 데이터 생성
    data, labels = make_classification(n_samples=20000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    return data, labels

def split_data_with_overlap(data, labels, overlap_ratio=0.8):
    """
    데이터셋을 Shadow와 Target 모델에 일부 겹치도록 나누는 함수.

    Parameters:
    data (array): 입력 데이터.
    labels (array): 데이터 라벨.
    overlap_ratio (float): Shadow 데이터와 Target 데이터 사이의 겹침 비율.

    Returns:
    Tuple: (shadow_data_train, shadow_data_test, shadow_labels_train, shadow_labels_test),
           (target_data_train, target_data_test, target_labels_train, target_labels_test)
    """
    # 전체 데이터를 무작위로 섞고 나눈다
    shadow_data, target_data, shadow_labels, target_labels = train_test_split(data, labels, test_size=0.5, random_state=42)

    # 중복 데이터 수 계산
    overlap_size = int(len(shadow_data) * overlap_ratio)

    # shadow_data에서 중복 데이터 추출
    overlap_data = shadow_data[:overlap_size]
    overlap_labels = shadow_labels[:overlap_size]

    # 나머지 데이터 설정
    shadow_data_rest = shadow_data[overlap_size:]
    shadow_labels_rest = shadow_labels[overlap_size:]

    # target_data에 중복 데이터 추가
    target_data = np.concatenate((target_data, overlap_data))
    target_labels = np.concatenate((target_labels, overlap_labels))

    # Shadow 모델 데이터를 추가로 Train/Test로 나눈다
    shadow_data_train, shadow_data_test, shadow_labels_train, shadow_labels_test = train_test_split(
        np.concatenate((shadow_data_rest, overlap_data)), 
        np.concatenate((shadow_labels_rest, overlap_labels)), 
        test_size=0.5, random_state=42
    )

    # 타겟 모델 데이터를 Train/Test로 나눈다
    target_data_train, target_data_test, target_labels_train, target_labels_test = train_test_split(
        target_data, target_labels, test_size=0.5, random_state=42)

    return (shadow_data_train, shadow_data_test, shadow_labels_train, shadow_labels_test), \
           (target_data_train, target_data_test, target_labels_train, target_labels_test)
