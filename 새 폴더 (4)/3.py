import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 📌 데이터 파일 경로
train_file = r'C:\Users\302-15\Desktop\새 폴더 (4)\open\train.csv'

# 📌 데이터 로드
train_data = pd.read_csv(train_file)

# ✅ 데이터 구조 확인
print(f"✅ Train Data Shape: {train_data.shape}")
print("🔍 Train Data Columns:", train_data.columns)

# 📌 ID 및 라벨 분리
X = train_data.iloc[:, 2:].values  # ID, label 제외
y = train_data.iloc[:, 1].values   # label만 추출

# 📌 데이터 크기 확인
num_samples, num_features = X.shape
img_size = int(np.sqrt(num_features))

print(f"✅ 샘플 개수: {num_samples}, 특징 개수: {num_features}")
print(f"✅ 이미지 크기 추정: {img_size}x{img_size}")

# 🚨 데이터 크기 검증
if img_size * img_size != num_features:
    print("🚨 데이터가 32x32 이미지 형식이 아닙니다!")
    print("⚠️ num_features 값이 1024(32x32)가 아닙니다. 데이터 구조 확인 필요.")
    exit()

# ✅ 데이터 변환
X = X.reshape(-1, img_size, img_size)  # 정규화 없이 원본 데이터 유지
print(f"✅ 데이터 리쉐이프 완료: {X.shape}")

# 📌 5개의 샘플 이미지 저장
num_samples_to_save = 5  # 저장할 이미지 개수
fig, axes = plt.subplots(1, num_samples_to_save, figsize=(15, 5))

for i in range(num_samples_to_save):
    sample_image = X[i]
    
    # 개별 이미지 저장
    plt.imsave(f"sample_image_{i}.png", sample_image, cmap='gray')
    
    # 플롯에 추가
    axes[i].imshow(sample_image, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f"Label: {y[i]}")

# 전체 이미지를 한 파일에 저장
plt.tight_layout()
plt.savefig("sample_images.png", dpi=300)
print("✅ 5개의 샘플 이미지 저장 완료: sample_images.png")
