from sklearn.datasets import make_classification
import numpy as np

# 첫 번째 데이터 생성
random_state = 42
data1, labels1 = make_classification(n_samples=20000, n_features=20, n_informative=15, n_redundant=5, random_state=random_state)

# 두 번째 데이터 생성 (같은 random_state 사용)
data2, labels2 = make_classification(n_samples=20000, n_features=20, n_informative=15, n_redundant=5, random_state=random_state)

# 두 데이터가 동일한지 확인
data_equal = np.array_equal(data1, data2)
labels_equal = np.array_equal(labels1, labels2)

print(data1)
print(data2)

print(f"Data is identical: {data_equal}")
print(f"Labels are identical: {labels_equal}")