import os
import torch
import numpy as np
import warnings
from skimage import color

_SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
_RESOURCE_DIR = os.path.join(_SOURCE_DIR, './resources')

def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error: Creating directory. ' + dir)

def get_resource_path(path):
   return os.path.join(_RESOURCE_DIR, path)

def lab_to_rgb(img):
    assert img.dtype == np.float32

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        return (255 * np.clip(color.lab2rgb(img), 0, 1)).astype(np.uint8)

def rgb_to_lab(img):
    #assert img.dtype == np.uint8

    return color.rgb2lab(img).astype(np.float32)

# save network
def save(ckpt_dir, net, optim, epoch):
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

	torch.save({
		'net': net.state_dict(), 'optim': optim.state_dict()},
		"%s/model_epoch%d.pth" % (ckpt_dir, epoch))


# load network
def load(ckpt_dir, net, optim, load_opt=None):
	if not os.path.exists(ckpt_dir):
		epoch = 0
		return net, optim, epoch

	if load_opt is None:
		ckpt_list = os.listdir(ckpt_dir)
		ckpt_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # 숫자만 이용하여 소팅
		epoch = int(ckpt_list[-1].split('epoch')[1].split('.pth')[0])
		dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_list[-1]))
	else:
		epoch = int(load_opt)
		dict_model = torch.load('%s/model_epoch%d.pth' % (ckpt_dir, epoch))

	net.load_state_dict(dict_model['net'])
	optim.load_state_dict(dict_model['optim'])


	return net, optim, epoch

