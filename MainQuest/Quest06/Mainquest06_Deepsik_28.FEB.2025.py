#!/usr/bin/env python
# coding: utf-8

# **모듈 불러오기

# In[7]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models, optimizers, regularizers

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pathlib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import shutil


# **1. 데이터분석

# In[18]:


# 1.2.No 증감 데이터 세트 

# 홈 디렉토리 경로를 확장
home = str(Path.home())

# 원본 데이터 경로와 출력 경로 설정
data_dir = os.path.join(home, 'aiffel/jellyfish')  # 원본 데이터 경로
output_dir = os.path.join(home, 'aiffel/jellyfish/no_aug')  # 증강되지 않은 데이터를 저장할 새 경로


# tf.data.Dataset 생성
dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    batch_size=32,
    image_size=(224, 224),
    shuffle=True
)


# 포함할 폴더 이름
use_folders = [
    "barrel",
    "blue",
    "compass",
    "lions_mane",
    "mauve_stinger",
    "moon"
]


# 클래스 확인
class_names = ["barrel",
    "blue",
    "compass",
    "lions_mane",
    "mauve_stinger",
    "moon"]    

# 각 클래스별 이미지 수를 저장할 딕셔너리
image_counts = {}

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 각 폴더(클래스) 순회
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)

    if os.path.isdir(class_path):  # 폴더인지 확인
        if class_name in use_folders:  # 포함할 폴더인지 확인
            # 새로운 클래스 폴더 생성
            new_class_path = os.path.join(output_dir, class_name)
            os.makedirs(new_class_path, exist_ok=True)

            # 파일 복사 ('aug'가 없는 파일만)
            for filename in os.listdir(class_path):
                if 'aug' not in filename:  # 'aug'가 없는 파일만 선택
                    src_path = os.path.join(class_path, filename)
                    dst_path = os.path.join(new_class_path, filename)
                    shutil.copy2(src_path, dst_path)
                    print(f"복사된 파일: {dst_path}")

            # 복사된 파일 수 계산
            copied_files = len([f for f in os.listdir(new_class_path) if os.path.isfile(os.path.join(new_class_path, f))])
            image_counts[class_name] = copied_files

# 결과 출력
print("\n각 클래스별 복사된 이미지 수:")
for class_name, count in image_counts.items():
    print(f"{class_name}: {count}개")

print(f"\n데이터가 다음 경로에 저장되었습니다: {output_dir}")
print("클래스 이름:", class_names)


# In[19]:


# 1.2.No 증감 데이터 세트 (2)

# 클래스별 이미지 수 계산
class_counts = {name: 0 for name in class_names}

# Dataset 생성
dataset_noaug = tf.keras.utils.image_dataset_from_directory(
    output_dir,
    batch_size=32,
    image_size=(224, 224),
    shuffle=True
)

# 데이터셋 순회하며 이미지 카운트
for images, labels in dataset_noaug:
    for label in labels:
        label_idx = int(label.numpy())
        if label_idx < len(class_names):
            class_name = class_names[label_idx]
            class_counts[class_name] += 1
        else:
            print(f"Warning: Label index {label_idx} is out of range")

# 결과 출력 및 시각화
print("\n클래스별 이미지 수:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}개")

# 시각화
plt.figure(figsize=(10, 6))
bars = plt.bar(class_counts.keys(), class_counts.values())
plt.title('Image distribution per class')
plt.xticks(rotation=45, ha='right')
plt.ylabel('no. of images')

# 각 바 위에 이미지 수 표시
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 통계 정보 출력
values = list(class_counts.values())
print("\n클래스별 데이터 통계:")
print(f"평균 이미지 수: {np.mean(values):.1f}")
print(f"최대 이미지 수: {np.max(values)}")
print(f"최소 이미지 수: {np.min(values)}")
print(f"표준 편차: {np.std(values):.1f}")
print(f"불균형 비율 (최대/최소): {np.max(values)/np.min(values):.2f}")


# In[20]:


# 1.3.증감 데이터 세트 (1)

# 배치 크기 정의
BATCH_SIZE = 32
# 이미지 높이와 너비 정의
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 1. 데이터세트를 numpy 배열로 변환
images_list = []
labels_list = []
for images, labels in dataset_noaug:
    images_list.append(images.numpy())
    labels_list.append(labels.numpy())

original_images = np.concatenate(images_list, axis=0)
original_labels = np.concatenate(labels_list, axis=0)

# 데이터 확인
print("이미지 배열 형태:", original_images.shape)
print("레이블 배열 형태:", original_labels.shape)

# Shape 확인
original_images = np.array(original_images)
original_labels = np.array(original_labels)
print("Original images shape:", original_images.shape)
print("Original labels shape:", original_labels.shape)

# 2. 학습세트와 나머지(검증+테스트)를 먼저 분리 (7:3)
X_train, X_temp, y_train, y_temp = train_test_split(
    original_images,
    original_labels,
    test_size=0.3,
    random_state=42,
    stratify=original_labels
)

# 3. 나머지(검증+테스트)를 다시 검증세트와 테스트세트로 균등 분할 (5:5)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# 4. 데이터 증강
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

# 5. 학습세트에 대한 데이터 증강 적용 (원본 데이터의 2배)
augmented_images = []
augmented_labels = []

# 학습 데이터에 대해서만 증강 적용
for image, label in zip(X_train, y_train):
    image = tf.convert_to_tensor(image)
    # 원본 이미지의 shape 유지를 위해 expand_dims 사용
    image = tf.expand_dims(image, 0)
    # 2개의 증강된 버전 생성
    for _ in range(2):
        aug_image = data_augmentation(image)
        augmented_images.append(aug_image[0].numpy())
        augmented_labels.append(label)

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

print("Augmented images shape:", augmented_images.shape)
print("Augmented labels shape:", augmented_labels.shape)

# 6. 원본과 증강 데이터 합치기
all_train_images = np.concatenate([X_train, augmented_images])
all_train_labels = np.concatenate([y_train, augmented_labels])

print("Final training images shape:", all_train_images.shape)
print("Final training labels shape:", all_train_labels.shape)

# 7. 최종 데이터세트 생성
augmented_train_dataset = tf.data.Dataset.from_tensor_slices(
    (all_train_images, all_train_labels)
).batch(BATCH_SIZE)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)
).batch(BATCH_SIZE)

# 8. 최종 데이터세트 크기 확인
print(f"\n최종 데이터세트 크기:")
print(f"원본 학습 이미지 크기: {len(X_train)}")
print(f"증감 학습 이미지 크기: {len(augmented_images)}")
print(f"학습세트 크기: {len(all_train_images)}")
print(f"검증세트 크기: {len(X_val)}")
print(f"테스트세트 크기: {len(X_test)}")


# In[26]:


# 1.3.증감 데이터 세트 (2)

## 모델에 사용할 데이터세트 시각화

# 클래스 분포 확인 및 시각화를 위한 데이터 반환
def check_distribution(dataset, dataset_name):
    labels = []
    for batch in dataset:
        _, label_batch = batch
        labels.extend(label_batch.numpy())
    
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip(unique, counts))
    
    print(f"\n{dataset_name} 클래스 분포:")
    for class_idx, count in dist.items():
        percentage = (count/len(labels)*100)
        class_name = class_names[class_idx]  # 클래스 이름 가져오기
        print(f"{class_name}: {count}개 ({percentage:.2f}%)")
    
    return dist

# 각 데이터셋의 분포 확인 및 데이터 저장
train_dist = check_distribution(augmented_train_dataset, "증강된 학습세트")
val_dist = check_distribution(validation_dataset, "검증세트")
test_dist = check_distribution(test_dataset, "테스트세트")

# 파이 차트 그리기
plt.style.use('default')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

def create_pie_chart(dist, ax, title):
    values = list(dist.values())
    labels = [class_names[i] for i in dist.keys()]
    total = sum(values)
    
    def make_autopct(values):
        def my_autopct(pct):
            val = int(round(pct*total/100.0))
            return f'{val}\n({pct:.1f}%)'
        return my_autopct
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(values)))    # tab20 컬러맵
    
    wedges, texts, autotexts = ax.pie(values, 
                                     labels=labels,
                                     colors=colors,
                                     autopct=make_autopct(values),
                                     startangle=90,
                                     labeldistance=0.7,    # 레이블 위치 조정 (1보다 작으면 안쪽)
                                     pctdistance=0.5,     # 퍼센트 위치 조정
                                     radius=1.0)           # 파이 차트 크기
        # 텍스트에 박스 추가
    for autotext in autotexts:
        autotext.set_bbox(dict(facecolor='white', 
                              alpha=0.7, 
                              edgecolor='none',
                              boxstyle='round,pad=0.2'))
        
    # 텍스트 스타일 설정
    plt.setp(autotexts, size=10, weight="bold")
    plt.setp(texts, size=11)
    
    ax.axis('equal')
    ax.set_title(title, pad=5)

# 각 데이터셋별 파이 차트 생성
create_pie_chart(train_dist, ax1, 'Augmented Training Set')
create_pie_chart(val_dist, ax2, 'Validation Set')
create_pie_chart(test_dist, ax3, 'Test Set')

print()
plt.tight_layout()
plt.show()


# **2. CV 모델: Resnet 18

# In[27]:


# 2-1. 모델 정의하기

def create_resnet18():
    def residual_block(x, filters, stride=1):
        # 스킵 연결을 위해 입력 저장
        shortcut = x

        # 첫 번째 컨볼루션 층
        x = layers.Conv2D(filters, 3, stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # 두 번째 컨볼루션 층
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        # 스킵 연결 처리
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # 스킵 연결 더하기
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)

        return x

    # 입력층
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # 초기 컨볼루션
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # ResNet 블록들
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)

    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)

    # 출력층
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# 모델 생성
model_resnet18 = create_resnet18()

# 모델 컴파일
model_resnet18.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 확인
model_resnet18.summary()

# 콜백 설정
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'resnet18_best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
]


# In[31]:


# Resnet 18 모델 레이어 분석 (1)

def get_layer_info(model):
    trainable_count = 0
    non_trainable_count = 0
    
    for layer in model.layers:
        print(f"Layer Name: {layer.name}")
        print(f"Layer Type: {type(layer).__name__}")
        print(f"Trainable Params: {layer.trainable}")
        print("----------------------")
        
        weights = layer.get_weights()
        if weights:
            print(f"Weight shapes: {[w.shape for w in weights]}")
        print("\n")
   
get_layer_info(model_resnet18)


# In[34]:


# Resnet 18 모델 레이어 분석 (2)

total_layers = len(model_resnet18.layers)
print(f"총 층의 개수: {total_layers}")

count = 0
for layer in model_resnet18.layers:
    count += 1
print(f"총 층의 개수: {count}")

# 모든 층의 이름과 함께 출력
for i, layer in enumerate(model_resnet18.layers):
    print(f"Layer {i+1}: {layer.name}")


# In[28]:


# Resnet 18 모델 학습
history_res18 = model_resnet18.fit(
    augmented_train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    callbacks=callbacks
)


# In[35]:


# 2-2. Resnet 18 학습 결과 시각화

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_res18.history['accuracy'], label='Training Accuracy')
plt.plot(history_res18.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_res18.history['loss'], label='Training Loss')
plt.plot(history_res18.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 테스트 데이터셋에서 성능 평가
test_loss, test_accuracy = model_resnet18.evaluate(test_dataset)
print(f"\n테스트 정확도: {test_accuracy:.4f}")

# 클래스별 성능 평가
predictions = []
true_labels = []

for images, labels in test_dataset:
    pred = model_resnet18.predict(images)
    predictions.extend(np.argmax(pred, axis=1))
    true_labels.extend(labels.numpy())

# 혼동 행렬 생성
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_labels, predictions)
print("\n혼동 행렬:")
print(cm)

# 분류 보고서 출력
print("\n분류 보고서:")
print(classification_report(true_labels, predictions, target_names=class_names))


# **3. CV Model: Resnet 50

# In[37]:


## Resnet 50 모델 구성하기

# 계층 모델 만들기
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_resnet50 = models.Sequential()
model_resnet50.add(base_model)
model_resnet50.add(layers.GlobalAveragePooling2D())
model_resnet50.add(layers.Dense(6, activation='softmax'))

# 모델 컴파일
model_resnet50.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 확인
model_resnet50.summary()

# 콜백 설정
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'resnet50_best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
]


# In[38]:


# Resnet 50 모델 레이어 분석 (1)

def get_layer_info(model):
    trainable_count = 0
    non_trainable_count = 0
    
    for layer in model.layers:
        print(f"Layer Name: {layer.name}")
        print(f"Layer Type: {type(layer).__name__}")
        print(f"Trainable Params: {layer.trainable}")
        print("----------------------")
        
        weights = layer.get_weights()
        if weights:
            print(f"Weight shapes: {[w.shape for w in weights]}")
        print("\n")
   
get_layer_info(model_resnet50)


# In[41]:


# Resnet 50 모델 레이어 분석 (2)

total_layers = len(model_resnet50.layers)
print(f"총 층의 개수: {total_layers}")

count = 0
for layer in model_resnet50.layers:
    count += 1
print(f"총 층의 개수: {count}")

# 모든 층의 이름과 함께 출력
for i, layer in enumerate(model_resnet50.layers):
    print(f"Layer {i+1}: {layer.name}")


# In[42]:


# Resnet 50 모델 학습

history_res50 = model_resnet50.fit(
    augmented_train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    callbacks=callbacks
)


# In[45]:


# Resnet 50 학습 결과 시각화

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_res50.history['accuracy'], label='Training Accuracy')
plt.plot(history_res50.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_res50.history['loss'], label='Training Loss')
plt.plot(history_res50.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 테스트 데이터셋에서 성능 평가
test_loss, test_accuracy = model_resnet50.evaluate(test_dataset)
print(f"\n테스트 정확도: {test_accuracy:.4f}")

# 클래스별 성능 평가
predictions = []
true_labels = []

for images, labels in test_dataset:
    pred = model_resnet50.predict(images)
    predictions.extend(np.argmax(pred, axis=1))
    true_labels.extend(labels.numpy())

# 혼동 행렬 생성
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_labels, predictions)
print("\n혼동 행렬:")
print(cm)

# 분류 보고서 출력
print("\n분류 보고서:")
print(classification_report(true_labels, predictions, target_names=class_names))


# **4. CV Model: DenseNet121

# In[69]:


## DenseNet 121 모델 구성

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)  # GAP 레이어로 벡터 크기 축소
x = Dense(512, activation='relu')(x)  # Fully Connected 레이어
predictions = Dense(6, activation='softmax')(x)  # 이진 분류용 출력 레이어

# 전이학습을 위해 기존 층들을 고정
base_model.trainable = False

# 전체 모델 생성
model_dense121 = Model(inputs=base_model.input, outputs=predictions)
model_dense121.summary()

for layer in base_model.layers:
    layer.trainable = False
    
model_dense121.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'densenet121_best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
]


# In[70]:


# Densenet 121 모델 레이어 분석 (1)

def get_layer_info(model):
    trainable_count = 0
    non_trainable_count = 0
    
    for layer in model.layers:
        print(f"Layer Name: {layer.name}")
        print(f"Layer Type: {type(layer).__name__}")
        print(f"Trainable Params: {layer.trainable}")
        print("----------------------")
        
        weights = layer.get_weights()
        if weights:
            print(f"Weight shapes: {[w.shape for w in weights]}")
        print("\n")
   
get_layer_info(model_dense121)


# In[71]:


# Densenet 121 모델 레이어 분석 (2)

total_layers = len(model_dense121.layers)
print(f"총 층의 개수: {total_layers}")

count = 0
for layer in model_dense121.layers:
    count += 1
print(f"총 층의 개수: {count}")

# 모든 층의 이름과 함께 출력
for i, layer in enumerate(model_dense121.layers):
    print(f"Layer {i+1}: {layer.name}")


# In[72]:


# 모델 학습
history_dense121 = model_dense121.fit(
    augmented_train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)


# In[74]:


# Densenet 121 학습 결과 시각화

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_dense121.history['accuracy'], label='Training Accuracy')
plt.plot(history_dense121.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_dense121.history['loss'], label='Training Loss')
plt.plot(history_dense121.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 테스트 데이터셋에서 성능 평가
test_loss, test_accuracy = model_dense121.evaluate(test_dataset)
print(f"\n테스트 정확도: {test_accuracy:.4f}")

# 클래스별 성능 평가
predictions = []
true_labels = []

for images, labels in test_dataset:
    pred = model_dense121.predict(images)
    predictions.extend(np.argmax(pred, axis=1))
    true_labels.extend(labels.numpy())

# 혼동 행렬 생성
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_labels, predictions)
print("\n혼동 행렬:")
print(cm)

# 분류 보고서 출력
print("\n분류 보고서:")
print(classification_report(true_labels, predictions, target_names=class_names))


# **5. CV Model: MobileNetV2

# In[21]:


## MobileNetV2 모델 구성

# 하이퍼파라미터 설정
NUM_CLASSES = 6
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

# MobileNetV2 모델 생성
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# 전이학습을 위해 기존 층들을 고정
base_model.trainable = False

# 모델 구성
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model_mnv2 = Model(inputs=base_model.input, outputs=predictions)

# 모델 컴파일
model_mnv2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 확인
model_mnv2.summary()

# 콜백 설정
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'mnv2_best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
]


# In[12]:


# MobileNetV2 모델 레이어 분석 (1)

def get_layer_info(model):
    trainable_count = 0
    non_trainable_count = 0
    
    for layer in model.layers:
        print(f"Layer Name: {layer.name}")
        print(f"Layer Type: {type(layer).__name__}")
        print(f"Trainable Params: {layer.trainable}")
        print("----------------------")
        
        weights = layer.get_weights()
        if weights:
            print(f"Weight shapes: {[w.shape for w in weights]}")
        print("\n")
   
get_layer_info(model_mnv2)


# In[22]:


# MobileNetV2 모델 레이어 분석 (2)

total_layers = len(model_mnv2.layers)
print(f"총 층의 개수: {total_layers}")

count = 0
for layer in model_mnv2.layers:
    count += 1
print(f"총 층의 개수: {count}")

# 모든 층의 이름과 함께 출력
for i, layer in enumerate(model_mnv2.layers):
    print(f"Layer {i+1}: {layer.name}")


# In[23]:


# MovileNetV2 모델 학습
history_mnv2 = model_mnv2.fit(
    augmented_train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    callbacks=callbacks
)


# In[24]:


# MobileNetV2 학습 결과 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_mnv2.history['accuracy'], label='Training Accuracy')
plt.plot(history_mnv2.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_mnv2.history['loss'], label='Training Loss')
plt.plot(history_mnv2.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 테스트 데이터셋에서 성능 평가
test_loss, test_accuracy = model_mnv2.evaluate(test_dataset)
print(f"\n테스트 정확도: {test_accuracy:.4f}")

# 클래스별 성능 평가
predictions = []
true_labels = []

for images, labels in test_dataset:
    pred = model_mnv2.predict(images)
    predictions.extend(np.argmax(pred, axis=1))
    true_labels.extend(labels.numpy())

# 혼동 행렬 생성
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_labels, predictions)
print("\n혼동 행렬:")
print(cm)

# 분류 보고서 출력
print("\n분류 보고서:")
print(classification_report(true_labels, predictions, target_names=class_names))


# **6. CV Model: MobileNetV2 (upgrade1 - Keggle Data 활용)

# In[5]:


# MovileNetV2 모델 학습 (Keggle Data 활용)

# 기본 ImageDataGenerator - 스케일링만 적용
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255  # 픽셀값을 0-1 범위로 정규화
)
# 하이퍼파라미터 설정
NUM_CLASSES = 6

# 데이터 로드
train_dir = '/aiffel/aiffel/jellyfish/Train_Test_Valid/train'
valid_dir = '/aiffel/aiffel/jellyfish/Train_Test_Valid/valid'
test_dir = '/aiffel/aiffel/jellyfish/Train_Test_Valid/test'

# 데이터 세트 만들기 
keggle_train_dataset = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),    # 언더스코어 제거
    batch_size=32,
    class_mode='categorical'
)

keggle_valid_dataset = datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

keggle_test_dataset = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)


# 데이터셋 정보 확인
print("\n=== 데이터셋 정보 ===")
print(f"클래스 정보: {keggle_train_dataset.class_indices}")
print(f"총 클래스 수: {len(keggle_train_dataset.class_indices)}")
print(f"훈련 데이터 수: {keggle_train_dataset.n}")
print(f"검증 데이터 수: {keggle_valid_dataset.n}")
print(f"테스트 데이터 수: {keggle_test_dataset.n}")

# 이후 모델 학습

model_mnv2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_mnv2_case1 = model_mnv2.fit(
    keggle_train_dataset,
    validation_data=keggle_valid_dataset,
    epochs=100,
    callbacks=callbacks
)


# In[6]:


# MobileNetV2 학습 결과 시각화

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_mnv2_case1.history['accuracy'], label='Training Accuracy')
plt.plot(history_mnv2_case1.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_mnv2_case1.history['loss'], label='Training Loss')
plt.plot(history_mnv2_case1.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 테스트 데이터셋에서 성능 평가
test_loss, test_accuracy = model_mnv2.evaluate(keggle_test_dataset)
print(f"\n테스트 정확도: {test_accuracy:.4f}")


# **7. CV Model: MobileNetV2 (upgrade2 - Google data 추가)

# https://www.notion.so/modulabs/1a75c8e5427d80d7bc5dcb84268047e7

# **8. CV Model: faster CNN + MobileNetV2 (combination)

# https://www.notion.so/modulabs/DLthon-1a55c8e5427d805faf4af33497f0490a?p=1a75c8e5427d80d7b792fa4978f99a83&pm=s
