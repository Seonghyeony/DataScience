import tensorflow as tf
from tensorflow import keras
from HandleInput import *
import numpy as np

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
    checkpoint_path = "./training_cnn/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_freq=5)
    model.fit(train_features, train_labels, epochs=10,
              validation_data=(val_features, val_labels),
              callbacks=[cp_callback])

def train_from_scratch():
    class_names = ['cat', 'cow', 'dog', 'pig', 'sheep']
    train_features, train_labels, test_features, test_labels = load_all_data()
    model = create_model()
    train(model, train_features, train_labels, test_features, test_labels)

    test_loss, test_acc = model.evaluate(test_features, test_labels)
    print('Test accuracy: ', test_acc)

    predictions = model.predict(test_features)
    print(predictions[0])
    print(np.argmax(predictions[0]))
    print(test_labels[0])

def load_weights_and_predict():
    model = create_model()
    model.load_weights('./training_cnn/cp-0020.ckpt')

    my_test_img = load_image('./test_image01.jpeg')
    my_test_img = np.reshape(my_test_img, [1, IMG_HEIGHT, IMG_WIDTH, 3])

    print(my_test_img.shape)
    my_prediction = model.predict(my_test_img)
    print(my_prediction[0])
    print(np.argmax(my_prediction[0]))


if __name__ == '__main__':
    train_from_scratch()
    # load_weights_and_predict()