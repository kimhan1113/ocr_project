"""Keras implementation of CRNN."""

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPool2D, BatchNormalization, LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers import RNN
from keras.layers import Reshape, Permute, Lambda
from keras.layers.advanced_activations import LeakyReLU


def CRNN(input_shape, num_classes, prediction_only=False, gru=True,training=1,bidirectional=True):
    """CRNN architecture.
    
    # Arguments
        input_shape: Shape of the input image, (256, 32, 1).
        num_classes: Number of characters in alphabet, including CTC blank.
        
    # References
        https://arxiv.org/abs/1507.05717
    """
    #K.clear_session()
    
    #act = LeakyReLU(alpha=0.3)
    act = 'relu'
    
    x = image_input = Input(shape=input_shape, name='image_input')
    print(input_shape) # batch x 256 x 32 x 1 
    x = Conv2D(64, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv1_1')(x)
    print(x) # batch x 256  x 32 x 64
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1', padding='same')(x)
    print(x) # batch x 128 x 16 x 64
    x = Conv2D(128, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv2_1')(x)
    print(x) # batch x 128 x 16 x 128
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool2', padding='same')(x)
    print(
x) # batch x 64 x 8 x 128
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv3_1')(x)
    print(x) # batch x 64 x 8 x 256
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv3_2')(x)
    print(x) # batch x 64 x 8 x 256
    x = MaxPool2D(pool_size=(2, 2), strides=(1, 2), name='pool3', padding='same')(x)
    print(x) # batch x 64 x 4 x 256
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv4_1')(x)
    print(x) # batch x 64 x 4 x 512
    x = BatchNormalization(name='batchnorm1')(x,training=training)
    print(x) # batch x 64 x 4 x 512
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv5_1')(x)
    print(x) # batch x 64 x 4 x 512
    x = BatchNormalization(name='batchnorm2')(x,training=training)
    print(x) # batch x 64 x 4 x 512
    x = MaxPool2D(pool_size=(2, 2), strides=(1, 2), name='pool5', padding='valid')(x)
    print(x) # batch x 63 x 2 x 512
    x = Conv2D(512, (2, 2), strides=(1, 1), activation=act, padding='valid', name='conv6_1')(x)
    print(x) # batch x 62 x 1 x 512
    x = Reshape((-1,512))(x)
    print(x) # batch x 62 x 512 --> 3 dimensional tensor


    if gru:
        x = Bidirectional(GRU(256, return_sequences=True))(x)
        x = Bidirectional(GRU(256, return_sequences=True))(x)
    else:
        if bidirectional :
            x = Bidirectional(LSTM(256, return_sequences=True))(x)
            x = Bidirectional(LSTM(256, return_sequences=True))(x)
        else : 
            x = LSTM(512, return_sequences=True)(x)
            x = LSTM(512, return_sequences=True)(x)
        # batch x 62 x 512
    print(x)
    x = Dense(num_classes, name='dense1')(x)
    print(x)
    x = y_pred = Activation('softmax', name='softmax')(x)
    print(x) # batch x 62 x 512
    
    model_pred = Model(image_input, x)
    
    if prediction_only:
        return model_pred

    max_string_len = int(y_pred.shape[1])

    # since keras doesn't currently support loss functions with extra parameters
    # CTC loss in lambda layer and dummy loss in compile call
    def ctc_lambda_func(args):
        labels, y_pred, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    labels = Input(name='label_input', shape=[max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])

    model_train = Model(inputs=[image_input, labels, input_length, label_length], outputs=ctc_loss)
    
    return model_train, model_pred
    
