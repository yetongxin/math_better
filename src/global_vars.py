test111 = 'hello world'
INPUT_LANG = None
OUTPUT_LANG = None

def set_input_lang(lang):
    global INPUT_LANG
    INPUT_LANG = lang

def get_input_lang():
    return INPUT_LANG

def set_output_lang(lang):
    global OUTPUT_LANG
    OUTPUT_LANG = lang

def get_output_lang():
    return OUTPUT_LANG


import torch_xla
import torch_xla.core.xla_model as xm
import turing
from turing.pytorch import ArnoldClusterResolver
print('turing version:', turing.__version__)

token = "9d8314cb972017ad3bcebaa7ae923c0a0a5a0536"
cluster = ArnoldClusterResolver(token=token, num_gpus=1)
xla = xm.xla_device()
print('Use device {}, its real device is {}'.format(xla, xm.xla_real_devices([xla])[0]))
device = xm.xla_device()

