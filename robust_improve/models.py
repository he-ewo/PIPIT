from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

def create_simple_model(input_shape=(20,)):
    # 간단한 모델 아키텍처 정의
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),  # 64 뉴런의 첫 번째 레이어
        Dense(32, activation='relu'),  # 32 뉴런의 두 번째 레이어
        Dense(1, activation='sigmoid')  # 출력 레이어
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_model(input_shape=(20,)):
    # 간단한 모델 아키텍처 정의
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),  # 64 뉴런의 첫 번째 레이어
        Dense(32, activation='relu'),  # 32 뉴런의 두 번째 레이어
        Dense(1, activation='sigmoid')  # 출력 레이어
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_complex_model(input_shape=(20,)):
    # 모델 아키텍처 정의
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),  # 256 뉴런
        BatchNormalization(),  # 배치 정규화
        Dropout(0.5),  # 드롭아웃
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # 128 뉴런
        BatchNormalization(),  # 배치 정규화
        Dropout(0.5),  # 드롭아웃
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # 64 뉴런
        BatchNormalization(),  # 배치 정규화
        Dropout(0.5),  # 드롭아웃
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),  # 32 뉴런
        BatchNormalization(),  # 배치 정규화
        Dense(1, activation='sigmoid')  # 출력 레이어
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
