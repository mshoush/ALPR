import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

plate_model_name = 'cnn/plates/plates-0.001-6conv-basic.model'
char_model_name = 'cnn/characters/characters-0.001-4conv-basic.model'

plate_img_size, char_img_size = 200, 50
lr = 1e-4

#Plate model
tf.reset_default_graph()

# CNN
convnet = input_data(shape=[None, plate_img_size, plate_img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu');  convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu');    convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

plate_model = tflearn.DNN(convnet, tensorboard_dir='log')
plate_model.load(plate_model_name)


# Characters model
tf.reset_default_graph()

# CNN
convnet = input_data(shape=[None, char_img_size, char_img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu');    convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 37, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

char_model = tflearn.DNN(convnet, tensorboard_dir='log')
char_model.load(char_model_name)
