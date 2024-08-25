import numpy as np
from data_processing import generate_data, split_data_with_overlap
from models import create_model, train_model, predict_model
from result import evaluate_membership_inference


# 데이터 생성 및 전처리
data, labels = generate_data()
(shadow_data_train, shadow_data_test, shadow_labels_train, shadow_labels_test), \
(target_data_train, target_data_test, target_labels_train, target_labels_test) = split_data_with_overlap(data, labels, overlap_ratio=0.2)

# Shadow 모델 학습
shadow_models = []
for i in range(3):  # 3개의 Shadow 모델 생성
    shadow_model = create_model()
    shadow_model = train_model(shadow_model, shadow_data_train, shadow_labels_train)
    shadow_models.append(shadow_model)

# Confidence Vector (예측 확률) 생성
shadow_train_preds = []
shadow_test_preds = []

for model in shadow_models:
    shadow_train_preds.append(predict_model(model, shadow_data_train))
    shadow_test_preds.append(predict_model(model, shadow_data_test))

# 공격 모델 학습 데이터 준비
x_attack = np.concatenate([np.concatenate(shadow_train_preds), np.concatenate(shadow_test_preds)])
y_attack = np.concatenate([np.ones(len(shadow_train_preds[0]))] * len(shadow_models) + 
                          [np.zeros(len(shadow_test_preds[0]))] * len(shadow_models))

# 공격 모델 학습
attack_model = create_model(input_shape=(1,))
attack_model = train_model(attack_model, x_attack, y_attack)

# 타겟 모델 생성 및 학습
target_model = create_model()
target_model = train_model(target_model, target_data_train, target_labels_train)

# 타겟 모델에 대한 멤버십 추론
shadow_dataset = np.concatenate([shadow_data_train, shadow_data_test])
target_preds = predict_model(target_model, shadow_dataset)

# 멤버십 추론을 위한 데이터 준비
x_target_attack = target_preds

# y_target_attack을 생성할 때 x_target_attack의 각 샘플이 target_data_train에 있는지 확인
# (no real-world) just for performance check
# we never know y_target_attack in realworld!!!
y_target_attack = []

for pred in shadow_dataset:
    if pred in target_data_train:
        y_target_attack.append(1)  # 타겟 모델의 학습 데이터에 포함된 경우
    else:
        y_target_attack.append(0)  # 타겟 모델의 학습 데이터에 포함되지 않은 경우

y_target_attack = np.array(y_target_attack)  # 0(포함x)과 1(포함o)로 이루어짐

# 멤버십 예측 수행
membership_predictions = predict_model(attack_model, x_target_attack)  # 해당 confidence vector가 훈련데이터셋어 있을 확률에 대한 예측

# 결과 출력 및 성능 지표 계산
evaluate_membership_inference(y_target_attack, membership_predictions)