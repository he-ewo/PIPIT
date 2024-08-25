from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model(input_shape=(20,)):
    # 모델 아키텍처 정의
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, data_train, labels_train, epochs=10):
    # 모델 학습
    model.fit(data_train, labels_train, epochs=epochs, verbose=1)
    return model

def predict_model(model, data):
    # 모델 예측
    return model.predict(data)
