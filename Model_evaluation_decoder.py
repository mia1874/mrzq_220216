import numpy as np
import scipy.io as scio
import tensorflow as tf
from Model_define_tf import NMSE, get_custom_objects

def Score(NMSE):
    score = (1 - NMSE) * 100
    return score

# Data loading
data_load_address = './data'
mat = scio.loadmat(data_load_address+'/Htest.mat')
x_test = mat['H_test']
x_test = x_test.astype('float32')

# load encoder_output
decode_input = np.load('./Modelsave/encoder_output.npy')

# load model and test NMSE
decoder_address = './Modelsave/decoder.h5'
_custom_objects = get_custom_objects()  # load keywords of Custom layers
model_decoder = tf.keras.models.load_model(decoder_address, custom_objects=_custom_objects)
y_test = model_decoder.predict(decode_input)
print('The NMSE is ' + np.str(NMSE(x_test, y_test)))

NMSE_test = NMSE(x_test, y_test)
scr = Score(NMSE_test)
if scr < 0:
    scr=0
else:
    scr=scr

result = 'score=', np.str(scr)
print(result)

