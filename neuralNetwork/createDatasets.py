# -*- encoding: utf -*-

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import transform as tf
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder

# %matplotlib inline

def create_captcha(text, shear=0, size=(100, 24)):
    """
    创建验证码字母图片数据
    :param text:
    :param shear:
    :param size:
    :return:
    """
    im = Image.new('L', size, 'black')
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(r'Coval.otf', 22)
    draw.text((2, 2), text, fill=1, font=font)
    image = np.array(im)
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    return image / image.max()

def segment_image(image):
    """
    将验证码图片分割成单个字母
    :param image:
    :return:
    """
    labeled_image = label(image > 0)
    subimages = []
    for region in regionprops(labeled_image):
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x, start_y:end_y])
    if len(subimages) == 0:
        return [image,]
    return subimages

random_state = check_random_state(14)
letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
shear_values = np.arange(0, 0.5, 0.05)
def generate_sample(random_state=None):
    """
    生成训练样本
    :param random_state:
    :return:
    """
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)
    return create_captcha(letter, shear=shear, size=(20, 20)), letters.index(letter)

def generate_datasets():
    datasets, targets = zip(*(generate_sample(random_state) for i in range(3000)))
    datasets = np.array(datasets, dtype='float')
    targets = np.array(targets)
    onehot = OneHotEncoder()
    y = onehot.fit_transform(targets.reshape(targets.shape[0], 1))
    y = y.todense()