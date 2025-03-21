import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parent / "open"

#  데이터 파일 경로
train_file = base_path / "train.csv"
test_file = base_path / "test.csv"
submission_file = base_path / "sample_submission.csv"


#  데이터 로드
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
submission = pd.read_csv(submission_file)

#  데이터 구조 확인
print(f" Train Data Shape: {train_data.shape}")
print(f" Test Data Shape: {test_data.shape}")
print(f" Sample Submission Shape: {submission.shape}")

# ID 컬럼 제거
train_data = train_data.iloc[:, 1:]  # 첫 번째 열 (ID) 제거

#  이미지 데이터 & 라벨 분리
X = train_data.iloc[:, 1:].values  # label 제외한 이미지 데이터
y = train_data.iloc[:, 0].values   # 첫 번째 열이 label (ID 제거했으므로)

#  데이터 크기 확인
num_samples, num_features = X.shape
img_size = int(np.sqrt(num_features))

print(f" 샘플 개수: {num_samples}, 특징 개수: {num_features}")
print(f" 이미지 크기 추정: {img_size}x{img_size}")

#  데이터 크기 검증
if img_size * img_size != num_features:
    print(" 데이터가 32x32 이미지 형식이 아닙니다!")
    print(" num_features 값이 1024(32x32)가 아닙니다. 데이터 구조 확인 필요.")
    exit()

#  데이터 변환
X = X.reshape(-1, img_size, img_size)  # 정규화 전 이미지 보기 위해 일단 2D로 리쉐이프
print(f" 데이터 리쉐이프 완료: {X.shape}")

#  샘플 이미지 시각화 (랜덤 5개)
fig, axes = plt.subplots(1, 5, figsize=(12, 6))
sample_indices = np.random.choice(num_samples, 5, replace=False)

for i, idx in enumerate(sample_indices):
    axes[i].imshow(X[idx], cmap="gray")
    axes[i].set_title(f"Label: {y[idx]}")
    axes[i].axis("off")

plt.show()

#  최종 데이터 변환 (모델 입력용)
X = X.reshape(-1, img_size, img_size, 1) / 255.0  # 정규화 (0~1)

#  학습 / 검증 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#  TensorFlow Dataset 변환
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).shuffle(1000)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

print(" 데이터 로드 및 변환 완료")
