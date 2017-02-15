from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import matplotlib.cm as cm
import PIL.Image as Image
from scipy.misc import imresize


def create_jpeg(img, output_filename):
    jet = cm.get_cmap('jet')
    mn = np.min(img)
    mx = np.max(img)
    img = img - mn
    img /= (mx-mn)
    img = np.maximum(0.0, np.minimum(1.0, img))
    img = jet(img)[:,:,0:3]
    img *= 255.0
    img = np.uint8(img)
    img = Image.fromarray(img)
    img.save(output_filename)
    print("saved %s" % output_filename)

def is_closed_five_point_box(points):
    assert len(points)==5
    ptA = points[0]
    # need to finish
    return True

def prep_img_for_vgg16(img, mean_to_subtract=None, interp='lanczos'):
    # resize options are here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imresize.html
    dest = np.empty((224,224,3), dtype=np.float32)

    if img.dtype != np.float32:
        img = img.astype(np.float32)

    assert len(img.shape)==2

    if img.shape != (224, 224):
        img = imresize(img, (224,224), interp=interp, mode='F')

    orig_resized_mean = np.mean(img)

    if mean_to_subtract is None:
        mean_to_subtract = orig_resized_mean

    img -= mean_to_subtract

    for ch in range(3):
        dest[:,:,ch]=img[:,:]
        
    dest = np.expand_dims(dest, axis=0)
    return dest, orig_resized_mean
