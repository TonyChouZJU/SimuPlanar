'''
Additional Data Transformation for Training
randomly choose an order for the three manipulations
and then choose a number between 0.5 and 1.5 for the
amount of enhancement
Finally, we then add the random lighting noise.

'''

#coding=utf-8
import _init_paths
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
from third_party.ImageFilterExtension import *
import subprocess
import cv2
import sys

print(__doc__)

class WLImageFilter():

    def __init__(self):
        #self.raw_img = img
        #self.enhanced_img = Image.fromarray(img.copy().astype(np.uint8))
        #self.filtered_img = img.copy()
        self.enhance_funcs = [ImageEnhance.Color, ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Sharpness]
        '''
        self.filter_funcs = [aqua, colorize, comic, aqua, colorize,
                             comic, darkness, diffuse,emboss,find_edge,
                             glowing_edge, ice, lighting, moire_fringe,
                             molten, mosaic, pencil, relief, sepia, sketch,
                             solarize, subtense, wave]
        '''
        #sepia,
        self.filter_funcs = [darkness, diffuse, lighting, moire_fringe, mosaic, relief,  wave]
        self.im_blur_funcs = [imMagic_radial_blur, imMagic_radial_blur_swirl, imMagic_motion_blur]

    #
    def enhance_img(self, raw_img):
        raw_img = Image.fromarray(raw_img.astype(np.uint8))
        #enhance_ratio = round(float(random.uniform(0.5, 1.5)), 2)
        enhance_ratio = np.random.uniform(0, 4)
        enhance_order = np.random.randint(len(self.enhance_funcs), size=1)
        called_func = self.enhance_funcs[enhance_order[0]]
        print called_func.__name__
        enhancer = called_func(raw_img)
        enhanced_img = enhancer.enhance(enhance_ratio)

        #blur_ratio = round(float(random.uniform(0.0, 0.8)), 2)
        blur_ratio = np.random.uniform(0, 4)
        enhanced_img = enhanced_img.filter(ImageFilter.GaussianBlur(blur_ratio))
        return np.array(enhanced_img).astype(np.uint8)

    def filter_img(self, raw_img, filter_nums=1):
        filtered_img = raw_img.astype(np.uint32)
        filter_order = np.random.randint(len(self.filter_funcs), size=filter_nums)
        for ind in range(len(filter_order)):
            called_func = self.filter_funcs[filter_order[ind]]
            print called_func.__name__
            filtered_img = called_func(filtered_img).astype(np.uint8)
        return filtered_img.astype(np.uint8)

    def im_blur_img(self, raw_img, filter_nums=1):
        filtered_img = raw_img.astype(np.uint32)
        filter_order = np.random.randint(len(self.im_blur_funcs), size=filter_nums)
        for ind in range(len(filter_order)):
            called_func = self.im_blur_funcs[filter_order[ind]]
            print called_func.__name__
            filtered_img = called_func(filtered_img).astype(np.uint8)
        return filtered_img.astype(np.uint8)


def imMagic_radial_blur(raw_img):
    raw_img = Image.fromarray(raw_img.astype(np.uint8))
    src_file = '/tmp/src.jpg'
    func_name = sys._getframe().f_code.co_name
    result_file = '/tmp/' + func_name + '.jpg'
    raw_img.save(src_file, 'JPEG')
    blur_ratio = str(np.random.uniform(0, 2))
    params = ['convert', src_file, '-radial-blur', blur_ratio, result_file]
    subprocess.check_call(params)
    result_img = cv2.imread(result_file)[:, :, (2, 1, 0)].astype(np.uint8)
    return result_img


def imMagic_radial_blur_swirl(raw_img):
    raw_img = Image.fromarray(raw_img.astype(np.uint8))
    src_file = '/tmp/src.jpg'
    func_name = sys._getframe().f_code.co_name
    result_file = '/tmp/' + func_name + '.jpg'
    raw_img.save(src_file, 'JPEG')
    blur_ratio = np.random.uniform(0, 2)
    swirle_ratio = np.random.uniform(-30, 30)
    params = ['convert', src_file, '-radial-blur', str(blur_ratio), '-swirl', str(swirle_ratio), result_file]
    subprocess.check_call(params)
    result_img = cv2.imread(result_file)[:, :, (2, 1, 0)].astype(np.uint8)
    return result_img


def imMagic_motion_blur(raw_img):
    raw_img = Image.fromarray(raw_img.astype(np.uint8))
    src_file = '/tmp/src.jpg'
    func_name = sys._getframe().f_code.co_name
    result_file = '/tmp/' + func_name + '.jpg'
    raw_img.save(src_file, 'JPEG')
    motion_ratio = np.random.uniform(10, 20)
    angle_ratio = np.random.uniform(-45, 45)
    if angle_ratio > 0:
        angle_ratio = '+' + str(angle_ratio)
    else:
        angle_ratio = '-' + str(angle_ratio)
    params = ['convert', src_file, '-motion-blur', '0x'+str(motion_ratio)+angle_ratio, result_file]
    subprocess.check_call(params)
    result_img = cv2.imread(result_file)[:, :, (2, 1, 0)].astype(np.uint8)
    return result_img
