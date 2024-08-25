import numpy as np
from data_processing import generate_data, split_data_with_overlap
from models import create_simple_model, create_complex_model, train_model, predict_model
from result import evaluate_membership_inference
from sklearn.metrics import accuracy_score
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential


# 데이터 생성 및 전처리
data, labels = generate_data()
(shadow_data_train, shadow_data_test, shadow_labels_train, shadow_labels_test), \
(target_data_train, target_data_test, target_labels_train, target_labels_test) = split_data_with_overlap(data, labels, overlap_ratio=0.2)

# Shadow 모델 학습
shadow_models = []
shadow_model_accuracies = []

for i in range(3):  # 3개의 Shadow 모델 생성
    shadow_model = create_simple_model()
    shadow_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    shadow_model = train_model(shadow_model, shadow_data_train, shadow_labels_train)
    shadow_models.append(shadow_model)

    # 각 Shadow 모델의 정확도 계산
    shadow_accuracy = accuracy_score(shadow_labels_test, (predict_model(shadow_model, shadow_data_test) > 0.5).astype(int))
    shadow_model_accuracies.append(shadow_accuracy)

# 가장 우수한 Shadow 모델 선택
best_shadow_model_index = np.argmax(shadow_model_accuracies)
best_shadow_model = shadow_models[best_shadow_model_index]

# 공격 모델 학습 데이터 준비 - 우수한 Shadow 모델의 결과만 사용
best_train_preds = predict_model(best_shadow_model, shadow_data_train)
best_test_preds = predict_model(best_shadow_model, shadow_data_test)

x_attack_best = np.concatenate([best_train_preds, best_test_preds])
y_attack_best = np.concatenate([np.ones(len(best_train_preds)), np.zeros(len(best_test_preds))])

# 공격 모델 학습 - 우수한 Shadow 모델의 결과만 사용
attack_model_best = Sequential([
    Input(shape=(1,)),
    Dense(1, activation='sigmoid')
])
attack_model_best.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
attack_model_best = train_model(attack_model_best, x_attack_best, y_attack_best)

# 타겟 모델 생성 및 학습 (Adversarial Regularization 적용 전)
target_model = create_simple_model()
target_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
target_model = train_model(target_model, target_data_train, target_labels_train)

# 타겟 모델에 대한 멤버십 추론 (Adversarial Regularization 적용 전)
shadow_dataset = np.concatenate([shadow_data_train, shadow_data_test])
target_preds = predict_model(target_model, shadow_dataset)

# 멤버십 추론을 위한 데이터 준비
x_target_attack = target_preds

# y_target_attack을 생성할 때 x_target_attack의 각 샘플이 target_data_train에 있는지 확인
y_target_attack = []

for pred in shadow_dataset:
    if pred in target_data_train:
        y_target_attack.append(1)  # 타겟 모델의 학습 데이터에 포함된 경우
    else:
        y_target_attack.append(0)  # 타겟 모델의 학습 데이터에 포함되지 않은 경우

y_target_attack = np.array(y_target_attack)

# 멤버십 예측 수행 - 우수한 Shadow 모델 결과만 사용 (Adversarial Regularization 적용 전)
membership_predictions_best = predict_model(attack_model_best, x_target_attack)

# 결과 출력 및 성능 지표 계산 - 우수한 Shadow 모델 결과만 사용 (Adversarial Regularization 적용 전)
print("\nResults using the Best Shadow Model (Before Robustness Improvement):")
evaluate_membership_inference(y_target_attack, membership_predictions_best)


# 새 타겟 모델 생성 및 학습 (Adversarial Regularization 적용 후)
adversarial_target_model = create_complex_model()
adversarial_target_model = train_model(adversarial_target_model, target_data_train, target_labels_train)

# 타겟 모델에 대한 멤버십 추론 (Adversarial Regularization 적용 후)
adversarial_target_preds = predict_model(adversarial_target_model, shadow_dataset)

# 멤버십 추론을 위한 데이터 준비 (Adversarial Regularization 적용 후)
x_adversarial_target_attack = adversarial_target_preds

# 멤버십 예측 수행 - 우수한 Shadow 모델 결과만 사용 (Adversarial Regularization 적용 후)
membership_predictions_adversarial = predict_model(attack_model_best, x_adversarial_target_attack)

# 결과 출력 및 성능 지표 계산 - 우수한 Shadow 모델 결과만 사용 (Adversarial Regularization 적용 후)
print("\nResults using the Best Shadow Model (After Robustness Improvement):")
evaluate_membership_inference(y_target_attack, membership_predictions_adversarial)
