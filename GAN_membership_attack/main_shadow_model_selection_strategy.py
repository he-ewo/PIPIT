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

for i in range(5):  # 3개의 Shadow 모델 생성->개수 바꿀 수 있음
    shadow_model = create_model()
    shadow_model = train_model(shadow_model, shadow_data_train, shadow_labels_train)
    shadow_models.append(shadow_model)

    # 각 Shadow 모델의 정확도 계산
    shadow_accuracy = accuracy_score(shadow_labels_test, (predict_model(shadow_model, shadow_data_test) > 0.5).astype(int))
    shadow_model_accuracies.append(shadow_accuracy)

# Confidence Vector (예측 확률) 생성 - 모든 Shadow 모델의 결과 사용
shadow_train_preds = []
shadow_test_preds = []

for model in shadow_models:
    shadow_train_preds.append(predict_model(model, shadow_data_train))
    shadow_test_preds.append(predict_model(model, shadow_data_test))

# 공격 모델 학습 데이터 준비 - 모든 Shadow 모델의 결과 사용
x_attack_all = np.concatenate([np.concatenate(shadow_train_preds), np.concatenate(shadow_test_preds)])
y_attack_all = np.concatenate([np.ones(len(shadow_train_preds[0]))] * len(shadow_models) + 
                              [np.zeros(len(shadow_test_preds[0]))] * len(shadow_models))

# 가장 우수한 Shadow 모델 선택
best_shadow_model_index = np.argmax(shadow_model_accuracies)
best_shadow_model = shadow_models[best_shadow_model_index]

# 공격 모델 학습 데이터 준비 - 우수한 Shadow 모델의 결과만 사용
best_train_preds = predict_model(best_shadow_model, shadow_data_train)
best_test_preds = predict_model(best_shadow_model, shadow_data_test)

x_attack_best = np.concatenate([best_train_preds, best_test_preds])
y_attack_best = np.concatenate([np.ones(len(best_train_preds)), np.zeros(len(best_test_preds))])



# 공격 모델 학습 - 모든 Shadow 모델의 결과 사용
attack_model_all = create_model(input_shape=(1,))
attack_model_all = train_model(attack_model_all, x_attack_all, y_attack_all)

# 공격 모델 학습 - 우수한 Shadow 모델의 결과만 사용
attack_model_best = create_model(input_shape=(1,))
attack_model_best = train_model(attack_model_best, x_attack_best, y_attack_best)



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

y_target_attack = np.array(y_target_attack)



# 멤버십 예측 수행 - 모든 Shadow 모델 결과 사용
membership_predictions_all = predict_model(attack_model_all, x_target_attack)

# 멤버십 예측 수행 - 우수한 Shadow 모델 결과만 사용
membership_predictions_best = predict_model(attack_model_best, x_target_attack)



# 결과 출력 및 성능 지표 계산 - 모든 Shadow 모델 결과 사용
print("Results using all Shadow Models:")
evaluate_membership_inference(y_target_attack, membership_predictions_all)

# 결과 출력 및 성능 지표 계산 - 우수한 Shadow 모델 결과만 사용
print("\nResults using the Best Shadow Model:")
evaluate_membership_inference(y_target_attack, membership_predictions_best)
