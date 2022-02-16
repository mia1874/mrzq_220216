import numpy as np
import scipy.io as scio
import tensorflow as tf
from Model_define_tf import get_custom_objects

# Data loading
data_load_address = './data'
mat = scio.loadmat(data_load_address+'/Htest.mat')
x_test = mat['H_test']
x_test = x_test.astype('float32')

# load model
encoder_address = './Modelsave/encoder.h5'
_custom_objects = get_custom_objects()  # load keywords of Custom layers
model_encoder = tf.keras.models.load_model(encoder_address, custom_objects=_custom_objects)
encode_feature = model_encoder.predict(x_test)
print("feedbackbits length is ", np.shape(encode_feature)[-1])
np.save('./Modelsave/encoder_output.npy', encode_feature)
