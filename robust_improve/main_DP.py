import numpy as np
from data_processing import generate_data, split_data_with_overlap
from models import create_model, train_model, predict_model
from result import evaluate_membership_inference
from sklearn.metrics import accuracy_score

# 데이터 생성 및 전처리
data, labels = generate_data()
(shadow_data_train, shadow_data_test, shadow_labels_train, shadow_labels_test), \
(target_data_train, target_data_test, target_labels_train, target_labels_test) = split_data_with_overlap(data, labels, overlap_ratio=0.2)

# Shadow 모델 학습
shadow_models = []
shadow_model_accuracies = []

for i in range(3):  # 3개의 Shadow 모델 생성
    shadow_model = create_model()
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
attack_model_best = create_model(input_shape=(1,))
attack_model_best = train_model(attack_model_best, x_attack_best, y_attack_best)

# 1. Baseline: 노이즈 없는 타겟 데이터
print("\n--- Baseline (No Noise) ---")

# 타겟 모델 생성 및 학습 (노이즈 없음)
target_model = create_model()
target_model = train_model(target_model, target_data_train, target_labels_train)

# 타겟 모델에 대한 멤버십 추론 (노이즈 없음)
shadow_dataset = np.concatenate([shadow_data_train, shadow_data_test])
target_preds = predict_model(target_model, shadow_dataset)

# 멤버십 추론을 위한 데이터 준비 (노이즈 없음)
x_target_attack = target_preds

y_target_attack = []
for pred in shadow_dataset:
    if pred in target_data_train:
        y_target_attack.append(1)  # 타겟 모델의 학습 데이터에 포함된 경우
    else:
        y_target_attack.append(0)  # 타겟 모델의 학습 데이터에 포함되지 않은 경우
y_target_attack = np.array(y_target_attack)

# 멤버십 예측 수행 (노이즈 없음)
membership_predictions_baseline = predict_model(attack_model_best, x_target_attack)

# 결과 출력 (노이즈 없음)
print("\nResults using the Best Shadow Model (No Noise):")
evaluate_membership_inference(y_target_attack, membership_predictions_baseline)


# 2. 노이즈가 추가된 타겟 데이터
print("\n--- Noisy Target Data ---")

# 타겟 데이터셋에 노이즈 추가 함수
def add_noise_to_dataset(dataset, noise_factor=0.3):   # 노이즈 어느 정도 넣을지 세팅가능
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=dataset.shape)
    return dataset + noise

# 노이즈를 추가한 타겟 데이터셋
noisy_target_data_train = add_noise_to_dataset(target_data_train)
noisy_target_data_test = add_noise_to_dataset(target_data_test)

# 노이즈가 추가된 타겟 모델 생성 및 학습
noisy_target_model = create_model()
noisy_target_model = train_model(noisy_target_model, noisy_target_data_train, target_labels_train)

# 타겟 모델에 대한 멤버십 추론 (노이즈 추가됨)
noisy_target_preds = predict_model(noisy_target_model, shadow_dataset)

# 멤버십 추론을 위한 데이터 준비 (노이즈 추가됨)
x_target_attack_noisy = noisy_target_preds

y_target_attack = []
for pred in shadow_dataset:
    if pred in target_data_train:
        y_target_attack.append(1)  # 타겟 모델의 학습 데이터에 포함된 경우
    else:
        y_target_attack.append(0)  # 타겟 모델의 학습 데이터에 포함되지 않은 경우
y_target_attack = np.array(y_target_attack)

# 멤버십 예측 수행 (노이즈 추가됨)
membership_predictions_noisy = predict_model(attack_model_best, x_target_attack_noisy)

# 결과 출력 (노이즈 추가됨)
print("\nResults using the Best Shadow Model (Noisy Target Data):")
evaluate_membership_inference(y_target_attack, membership_predictions_noisy)
