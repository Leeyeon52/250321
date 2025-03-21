import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# 데이터 로드
train_data = pd.read_csv("open/train.csv")
test_data = pd.read_csv("open/test.csv")
submission = pd.read_csv("open/sample_submission.csv")

# 데이터 구조 확인
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# 데이터 전처리
X = train_data.iloc[:, 1:].values
y = train_data.iloc[:, 0].values

num_features = X.shape[1]

# 이미지 크기를 자동 설정
if num_features == 1024:
    img_size = 32
elif num_features == 784:
    img_size = 28
else:
    raise ValueError(f"❌ 데이터 크기가 맞지 않습니다! 예상치 못한 num_features: {num_features}")

print(f"자동 설정된 이미지 크기: {img_size}x{img_size}")

# 데이터 변환
X = X.reshape(-1, img_size, img_size, 1) / 255.0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터셋 생성
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).shuffle(1000)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# 모델 생성
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 모델 컴파일 및 학습
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# 테스트 데이터 변환
X_test = test_data.iloc[:, 1:].values
X_test = X_test.reshape(-1, img_size, img_size, 1) / 255.0

# 예측
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# 결과 저장
submission['label'] = y_pred_labels
submission.to_csv('./baseline_submission.csv', index=False, encoding='utf-8-sig')

print("✅ 예측 완료! 결과를 baseline_submission.csv에 저장했습니다.")
