import numpy as np
from data_processing import generate_data, split_data_with_overlap
from models import create_model, train_model, predict_model
from result import evaluate_membership_inference
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from keras.optimizers import Adam
from keras.models import Model

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
def add_noise_to_dataset(dataset, noise_factor=0.1):
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


# 3. 노이즈 추가된 데이터에 GAN을 이용한 데이터 증강
print("\n--- Noisy + GAN-Augmented Target Data ---")

# GAN 생성자 모델 생성
def build_generator(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(input_dim, activation='tanh'))
    return model

# GAN 판별자 모델 생성
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# GAN 모델 생성 및 학습
def train_gan(generator, discriminator, data, epochs=500, batch_size=64, sample_interval=1000):
    # GAN 컴파일
    optimizer = Adam(0.0002, 0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    discriminator.trainable = False

    z = Input(shape=(data.shape[1],))
    img = generator(z)
    validity = discriminator(img)
    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    # GAN 학습
    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]

        noise = np.random.normal(0, 1, (batch_size, data.shape[1]))
        gen_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, data.shape[1]))
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")

print('####################################################################')


# GAN 모델 구성
generator = build_generator(input_dim=noisy_target_data_train.shape[1])
discriminator = build_discriminator(input_dim=noisy_target_data_train.shape[1])

# GAN 학습
train_gan(generator, discriminator, noisy_target_data_train)

# GAN으로 증강된 데이터 생성
noise = np.random.normal(0, 1, (noisy_target_data_train.shape[0], noisy_target_data_train.shape[1]))
gan_augmented_data = generator.predict(noise)

# GAN으로 증강된 데이터와 노이즈 데이터 결합
augmented_target_data_train = np.concatenate([noisy_target_data_train, gan_augmented_data])
augmented_target_labels_train = np.concatenate([target_labels_train, target_labels_train])

# 증강된 타겟 모델 생성 및 학습
augmented_target_model = create_model()
augmented_target_model = train_model(augmented_target_model, augmented_target_data_train, augmented_target_labels_train)

# 타겟 모델에 대한 멤버십 추론 (증강된 데이터)
augmented_target_preds = predict_model(augmented_target_model, shadow_dataset)

# 멤버십 추론을 위한 데이터 준비 (증강된 데이터)
x_target_attack_augmented = augmented_target_preds

y_target_attack = []
for pred in shadow_dataset:
    if pred in target_data_train:
        y_target_attack.append(1)  # 타겟 모델의 학습 데이터에 포함된 경우
    else:
        y_target_attack.append(0)  # 타겟 모델의 학습 데이터에 포함되지 않은 경우
y_target_attack = np.array(y_target_attack)

# 멤버십 예측 수행 (증강된 데이터)
membership_predictions_augmented = predict_model(attack_model_best, x_target_attack_augmented)

# 결과 출력 (증강된 데이터)
print("\nResults using the Best Shadow Model (Noisy + GAN-Augmented Target Data):")
evaluate_membership_inference(y_target_attack, membership_predictions_augmented)
