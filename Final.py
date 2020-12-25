import tensorflow as tf
import os
import glob
import cv2
import numpy as np
from pathlib import Path
from tensorflow import keras

import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

IMG_HEIGHT = 40
IMG_WIDTH = 60
NUM_CHANNEL = 3
NUM_CLASS = 2
IMAGE_DIR_BASE = './Oxford-IIIT_Pet_Dataset'

def test_if_valid_jpeg(path):
  img = tf.io.read_file(path)
  image = bytearray(img.numpy())
  if image[0] == 255 and image[1] == 216 and image[-2] == 255 and image[-1] == 217:
    return True
  else:
    return False

def load_image(addr):
    img = cv2.imread(addr)
    # 흑백 이미지일 경우
    # img = cv2.imread(addr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)  # 32비트 실수형으로 변환
    # img = img.ravel()   # 다차원 배열을 1차원 벡터로.
    img = (img - np.mean(img)) / np.std(img)  # 일종의 정규화

    return img

# 이미지 셔플 함수.
def image_shuffle(features, labels):
    idxs = np.arange(0, len(features))
    np.random.shuffle(idxs)
    features = np.array(features)
    labels = np.array(labels)
    return features[idxs], labels[idxs]

def load_data_set():
    # glob 유틸리티를 이용하여 images 디렉토리에 저장된 파일들의 경로명을 수집하고 알파벳 순으로 정렬.
    image_paths = glob.glob('{}/images/*.jpg'.format(IMAGE_DIR_BASE))
    image_paths.sort()
    print(image_paths)

    # 수집한 경로명들을 trainval.txt 파일과 test.txt 파일의 내용에 따라 분류.
    trainval_list = open('{}/annotations/trainval.txt'.format(IMAGE_DIR_BASE), 'r')
    Lines = trainval_list.readlines()
    train_file_name = [line.split()[0] for line in Lines]
    train_file_label = [line.split()[2] for line in Lines]

    test_list = open('{}/annotations/test.txt'.format(IMAGE_DIR_BASE), 'r')
    Lines2 = test_list.readlines()
    test_file_name = [line.split()[0] for line in Lines2]
    test_file_label = [line.split()[2] for line in Lines2]

    train_features = []   # empty Python list
    train_labels = []     # 0 cat, 1 dog
    test_features = []  # empty Python list
    test_labels = []  # 0 cat, 1 dog

    # train_image와 test_image에 해당하는 이미지들을 각각 나눈다.
    for image_path in image_paths:
        # 손상된 jpeg 파일이 있다면 건너뛴다.
        if not test_if_valid_jpeg(image_path):
            continue

        if Path(image_path).stem in train_file_name:
            image = load_image(image_path)
            train_features.append(image)
            train_labels.append(int(train_file_label[train_file_name.index(Path(image_path).stem)]) - 1)

        if Path(image_path).stem in test_file_name:
            image = load_image(image_path)
            test_features.append(image)
            test_labels.append(int(test_file_label[test_file_name.index(Path(image_path).stem)]) - 1)

    # print(len(train_features))
    # print(len(train_labels))
    # print(len(test_features))
    # print(len(test_labels))

    # 이미지들을 shuffle 한다.
    shuf_train_features, shuf_train_labels = image_shuffle(train_features, train_labels)
    shuf_test_features, shuf_test_labels = image_shuffle(test_features, test_labels)

    return shuf_train_features, shuf_train_labels, shuf_test_features, shuf_test_labels


def load_all_data():
    train_images, train_labels, test_images, test_labels = load_data_set()
    return train_images, train_labels, test_images, test_labels


def create_model():
    model = keras.Sequential([
        # 커널 32개, 커널 사이즈 3, 함수 relu, 패딩을 하겠다, 첫 번째 layer에 대해서만 input_shape을 줘야한다.
        keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL)),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(NUM_CLASS, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model

def train(model, train_features, train_labels, val_features, val_labels):
    checkpoint_path = "./training_final/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_freq=5)
    model.fit(train_features, train_labels, epochs=10,
              validation_data=(val_features, val_labels),
              callbacks=[cp_callback])

def train_from_scratch():
    train_features, train_labels, test_features, test_labels = load_all_data()
    model = create_model()
    train(model, train_features, train_labels, test_features, test_labels)

    # 모든 테스트 데이터에 대해 평균 정확도 계산 후 출력!
    test_loss, test_acc = model.evaluate(test_features, test_labels)
    print('Test accuracy: ', test_acc)

    # predictions = model.predict(test_features)
    # print(predictions[0])
    # print(np.argmax(predictions[0]))
    # print(test_labels[0])

def load_weights_and_predict():
    train_features, train_labels, test_features, test_labels = load_all_data()
    model = create_model()
    model.load_weights('./training_final/cp-0006.ckpt')

    # my_test_img = load_image('./test_image01.jpeg')
    # my_test_img = np.reshape(my_test_img, [1, IMG_HEIGHT, IMG_WIDTH, 3])
    #
    # print(my_test_img.shape)
    # my_prediction = model.predict(my_test_img)
    # print(my_prediction[0])
    # print(np.argmax(my_prediction[0]))

    # 모든 테스트 데이터에 대해 평균 정확도 계산 후 출력!
    test_loss, test_acc = model.evaluate(test_features, test_labels)
    print('Test average accuracy: ', test_acc)


if __name__ == '__main__':
    # train_from_scratch()
    load_weights_and_predict()