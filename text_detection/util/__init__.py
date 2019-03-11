import log
import dtype
import plt
import np
import img
_img = img
import dec
import rand
import mod
import proc
import test
import neighbour as nb
import str_ as str
import io_ as io
import feature
import thread_ as thread
import caffe_ as caffe
import tf
import cmd
import ml
import url
import time_ as time
from progress_bar import ProgressBar

init_logger = log.init_logger


def get_temp_path(name=''):
    path = 'result/%s_%s.png' % (log.get_date_str(), name[:-4])
    return path


def sit(img=None, name=''):

    path = get_temp_path(name)
        
    if img is None:
        plt.save_image(path)
        return path
    
    img = _img.bgr2rgb(img)
    if type(img) == list:
        print "hehe"
        plt.show_images(images = img, path = path, show = False, axis_off = True, save = True)
    else:
        plt.imwrite(path, img)
    return path


def get_count():
    global _count
    _count = 0
    _count += 1
    return _count    
