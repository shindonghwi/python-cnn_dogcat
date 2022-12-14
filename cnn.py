""" 1. Import Library """
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)
print(tf.config.list_physical_devices())

""" 2. Data Preprocessing"""
# Training data set preprocessing
# 훈련 과적합을 방지하기 위해 훈련 데이터넷을 전처리한다.
# * 과적합: 훈련 데이터셋에서는 정확도가 아주 높게 나오지만, 테스트 데이터셋에서는 정확도가 훨씬 낮은 현상
train_dategen = ImageDataGenerator(
    rescale=1. / 255,  # 픽셀 값을 255로 나누어서 스케일링 적용
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_dategen.flow_from_directory(
    directory='dataset/training_set',
    target_size=(64, 64),  # 이미지의 최종 크기 - 컨벌루션의 입력
    batch_size=32,
    class_mode='binary'
)

# Test data set preprocessing
test_dategen = ImageDataGenerator(
    rescale=1. / 255
)  # 테스트 이미지는 원본처럼 그대로 유지하여야한다. ( 새로들어온 이미지가 변형이 안되어 있을 거기에! )

test_set = test_dategen.flow_from_directory(
    directory='dataset/test_set',
    target_size=(64, 64),  # 이미지의 최종 크기 - 컨벌루션의 입력
    batch_size=32,
    class_mode='binary'
)

""" 3. Building CNN """
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

# cnn init
cnn = Sequential()

# 1 convolution & pooling ( max )
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(MaxPool2D(pool_size=(2, 2), strides=2))

# 2 convolution & pooling ( max )
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPool2D(pool_size=(2, 2), strides=2))

# 3 convolution & pooling ( max )
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPool2D(pool_size=(2, 2), strides=2))

# 4 Flattening
cnn.add(Flatten())

# 5 Full Connection
cnn.add(Dense(units=128, activation='relu'))

# 6 Output Layer
# 이중 분류(개,고양이) 에서는 시그모이드 활성화 함수를 권장
# 다중 분류에서는 소프트맥스 활성화 함수 사용 ( 각 예측 확률의 합을 1로 만들어야하기에 )
cnn.add(Dense(units=1, activation='sigmoid'))

""" 4. Training CNN """
# 1. compile
# adam: 확률적 경사 하강법
# binary_crossentropy: 이진분류를 하기에 바이너리 크로스엔트로피 설정
# metrics -> accuracy: 분류 모델의 성능을 측정하는 적절한 방법
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 2. train & evaluating test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

""" 5. Prediction Single """
import numpy as np
from keras.preprocessing import image

test_image_1 = image.image_utils.load_img(path='dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image_2 = image.image_utils.load_img(path='dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))

test_image_1 = image.image_utils.img_to_array(test_image_1)
test_image_2 = image.image_utils.img_to_array(test_image_2)

test_image_1 = np.expand_dims(test_image_1, axis=0)  # 배치 사이즈를 훈련떄와 맞춰야한다.
test_image_2 = np.expand_dims(test_image_2, axis=0)

result1 = cnn.predict(test_image_1 / 255.0)
result2 = cnn.predict(test_image_2 / 255.0)

print(training_set.class_indices)

if result1[0][0] > 0.5:  # 개
    prediction_1 = 'dog'
else:
    prediction_1 = 'cat'

if result2[0][0] > 0.5:  # 개
    prediction_2 = 'dog'
else:
    prediction_2 = 'cat'

print('result1: ', result1)
print('result2: ', result2)

print('-------------')

print('result_prediction_1: ', result1[0][0])
print('result_prediction_2: ', result2[0][0])
print('prediction_1: ', prediction_1)
print('prediction_2: ', prediction_2)
