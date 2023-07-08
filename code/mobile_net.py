'''
Fine tuning the MobileNet model for sign language digit recognization
By removing the last 6 layers and adding layers according to our requirments.
With changing the functional paradigm of the model
'''

import os
import tensorflow as tf
from utils import count_params
from keras import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from utils import plot_confusion
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

def create_model():
    model = tf.keras.applications.mobilenet.MobileNet()
    print("Mobile Net Model Summary")
    model.summary()

    x = model.layers[-6].output
    g = GlobalAveragePooling2D()(x)
    output = Dense(units=10,activation='softmax')(g)

    model = Model(inputs=model.input, outputs=output)
    for layer in model.layers[:-35]:
        layer.trainable = False
    print('Modified Model')
    model.summary()

    params = count_params(model)
    assert params['trainable_params'] == 2411530 
    assert params['non_trainable_params'] == 827584
    return model


def model_fit(model,train_batches, valid_batches):
    callback = [ReduceLROnPlateau(monitor = 'val_loss', patience = 10, factor=0.25, verbose=1),
                ModelCheckpoint("smobilenet.h5",save_best_only=True)]

    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy','loss'])

    model.fit(x=train_batches,
            validation_data=valid_batches,
            epochs=30,
            verbose=2,
            callbacks=callback)

    if os.path.isfile('smobilenet.h5') is False:
        model.save('smobilenet.h5')


def main():

    model = create_model()

    path = os.getcwd()
    train_path = path + '/sign_lang_digits/dataset/train' 
    test_path = path + '/sign_lang_digits/dataset/test' 
    valid_path = path + '/sign_lang_digits/dataset/valid' 

    preprocessing_function = tf.keras.applications.mobilenet.preprocess_input
    generator = ImageDataGenerator(preprocessing_function, featurewise_center=True,)
    
    train_batches = generator.flow_from_directory(directory=train_path,
                                                target_size=(224,224),
                                                batch_size=10)
    
    test_batches = generator.flow_from_directory(directory=test_path,
                                                target_size=(224,224),
                                                shuffle=False,
                                                batch_size=10)
    
    valid_batches = generator.flow_from_directory(directory=valid_path,
                                                target_size=(224,224),
                                                batch_size=10)

    test_labels = test_batches.classes

    history = model_fit(model,train_batches,valid_batches) 
    predictions = model.predict(x=test_batches, verbose=0)

    plot_training()
    y_pred=predictions.argmax(axis=1)
    plot_confusion(test_labels,y_pred)

if __name__ == '__main__':
    main()
