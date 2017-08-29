import numpy as np
from keras import models, layers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from util import load_train_data, load_test_data, save_predictions


def build_model():
    image = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(image)
    x = layers.MaxPool2D(pool_size=2, padding='valid')(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, padding='valid')(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense((7 * 7 * 32 + 10) // 2, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)

    prediction = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=image, outputs=prediction)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    return model


def main():
    model = build_model()
    model.summary()

    train_images, targets = load_train_data()
    train_images = train_images.reshape(-1, 28, 28, 1)
    targets = to_categorical(targets, 10)
    callbacks = [
        EarlyStopping(monitor='val_acc', patience=3),
        ModelCheckpoint('keras_cnn.h5', save_best_only=True, save_weights_only=True),
    ]
    model.fit(train_images, targets, batch_size=64, epochs=100, validation_split=0.1, callbacks=callbacks)

    model.load_weights('keras_cnn.h5')
    test_images = load_test_data()
    test_images = test_images.reshape(-1, 28, 28, 1)
    predictions = model.predict(test_images)
    labels = np.argmax(predictions, 1)
    save_predictions(labels, 'keras_cnn.csv')


if __name__ == '__main__':
    main()
