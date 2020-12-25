import tensorflow as tf
from tensorflow import keras
from HandleInput import *
import numpy as np

from matplotlib import pyplot as plt
print(tf.__version__)

class_names = ['cat', 'cow', 'dog', 'pig', 'sheep']
train_features, train_labels, test_features, test_labels = load_all_data()

# plt.figure()
# plt.imshow(train_features[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
#
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_features[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
#
# plt.show()


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL)),
        keras.layers.Dense(512, activation=tf.keras.activations.relu),
        keras.layers.Dense(NUM_CLASS, activation=tf.keras.activations.softmax)
    ])

    # 추가적으로 지정해주어야 할 것.
    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

checkpoint_path = "../training/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=5)

model = create_model()
model.summary()

# 원래는 validation_data 와 test_data를 다르게 설정하는 것이 정석적인 방법이다.
model.fit(train_features, train_labels, epochs=10,
          validation_data= (test_features, test_labels),
          callbacks= [cp_callback])

model.save_weights('../training/final_weight.ckpt')
# model.load_weights('../training/cp-0005.ckpt')


# 평가. test_acc: 적중률.
test_loss, test_acc = model.evaluate(test_features, test_labels)
print('Test accuracy: ', test_acc)

# 어떤걸 맞췄고, 어떤걸 틀렸고.. 이런 것을 알고 싶을 때 쓰는 함수.
# 각각의 클래스에 속할 확률을 리턴해준다.
predictions = model.predict(test_features)
print(predictions[0])
# argmax(): 최대값의 위치 인덱스를 찾는다.
print(np.argmax(predictions[0]))
# 정답을 출력.
print(test_labels[0])


# 학습이 완료된 신경망을 테스트 해봄.
my_test_img = load_image('../test_image01.jpeg')
# !! 이미지 크기를 맞춰주어야 한다.

# 4차원 배열로 맞춰줘야 함. 밑의 2가지 방법이 있다.
my_test_img = np.reshape(my_test_img, [1, IMG_HEIGHT, IMG_WIDTH, 3])
# my_test_img = (np.expand_dims(my_test_img, 0))

print(my_test_img.shape)
my_prediction = model.predict(my_test_img)
print(my_prediction[0])
print(np.argmax(my_prediction[0]))


















