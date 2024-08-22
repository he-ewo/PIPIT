import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 데이터 생성 (예제에서는 임의의 데이터 사용)
data = np.random.rand(1000, 20)  # 1000개의 샘플, 각 샘플은 20개의 특징으로 구성
labels = np.random.randint(2, size=(1000, 1))  # 0 또는 1의 이진 라벨

# Shadow 모델을 위한 데이터셋 분리
shadow_data_train, shadow_data_test, shadow_labels_train, shadow_labels_test = train_test_split(data, labels, test_size=0.5)

# Shadow 모델 아키텍처 정의 (타겟 모델과 동일한 구조)
def create_shadow_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(20,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Shadow 모델 학습
shadow_model_1 = create_shadow_model()
shadow_model_1.fit(shadow_data_train, shadow_labels_train, epochs=10, verbose=1)

shadow_model_2 = create_shadow_model()
shadow_model_2.fit(shadow_data_train, shadow_labels_train, epochs=10, verbose=1)

# Confidence Vector (예측 확률) 생성
shadow_train_preds_1 = shadow_model_1.predict(shadow_data_train)
shadow_test_preds_1 = shadow_model_1.predict(shadow_data_test)

shadow_train_preds_2 = shadow_model_2.predict(shadow_data_train)
shadow_test_preds_2 = shadow_model_2.predict(shadow_data_test)

# 공격 모델 학습 데이터 준비
x_attack = np.concatenate([shadow_train_preds_1, shadow_test_preds_1, shadow_train_preds_2, shadow_test_preds_2])
y_attack = np.concatenate([np.ones(len(shadow_train_preds_1)), np.zeros(len(shadow_test_preds_1)),
                           np.ones(len(shadow_train_preds_2)), np.zeros(len(shadow_test_preds_2))])

# 공격 모델 아키텍처 정의
attack_model = Sequential([
    Dense(128, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
attack_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 공격 모델 학습
attack_model.fit(x_attack, y_attack, epochs=10, verbose=1)

# 타겟 모델에 대한 멤버십 추론
# 여기서는 타겟 모델을 shadow_model_1로 가정
target_train_preds = shadow_model_1.predict(shadow_data_train)
target_test_preds = shadow_model_1.predict(shadow_data_test)

# 멤버십 추론을 위한 데이터 준비
x_target_attack = np.concatenate([target_train_preds, target_test_preds])
y_target_attack = np.concatenate([np.ones(len(target_train_preds)), np.zeros(len(target_test_preds))])

# 멤버십 예측 수행
membership_predictions = attack_model.predict(x_target_attack)

# 결과 출력
for i, pred in enumerate(membership_predictions):
    print(f"샘플 {i}: {'Train' if pred > 0.5 else 'Test'}")
