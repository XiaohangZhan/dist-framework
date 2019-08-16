import numpy as np
import sys
import cv2

from PIL import Image

def bbox_iou(b1, b2):
    '''
    b: (x1,y1,x2,y2)
    '''
    lx = max(b1[0], b2[0])
    rx = min(b1[2], b2[2])
    uy = max(b1[1], b2[1])
    dy = min(b1[3], b2[3])
    if rx <= lx or dy <= uy:
        return 0.
    else:
        interArea = (rx-lx)*(dy-uy)
        a1 = float((b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = float((b2[2] - b2[0]) * (b2[3] - b2[1]))
        return interArea / (a1 + a2 - interArea)
    
def crop_padding(img, roi, pad_value):
    '''
    Crop an image with arbitrary bbox, i.e., the bbox is allowed to be beyond the image borders. padded value can be specified.
    img: HxW or HxWxC np.ndarray
    roi: (x,y,w,h)
    pad_value: e.g., (0,0,0) for 3-channel image or (0,) for single channel image
    '''
    need_squeeze = False
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
        need_squeeze = True
    assert len(pad_value) == img.shape[2]
    x,y,w,h = roi
    x,y,w,h = int(x),int(y),int(w),int(h)
    H, W = img.shape[:2]
    output = np.tile(np.array(pad_value), (h, w, 1)).astype(img.dtype)
    if bbox_iou((x,y,x+w,y+h), (0,0,W,H)) > 0:
        output[max(-y,0):min(H-y,h), max(-x,0):min(W-x,w), :] = img[max(y,0):min(y+h,H), max(x,0):min(x+w,W), :]
    if need_squeeze:
        output = np.squeeze(output)
    return output


def image_resize(image, short_size=None, long_size=None):
    '''
    Resize image by specify short_size or long_size
    img: numpy.ndarray
    '''
    assert (short_size is None) ^ (long_size is None)
    h, w = image.shape[:2]
    if short_size is not None:
        if w < h:
            neww = short_size
            newh = int(short_size / float(w) * h)
        else:
            neww = int(short_size / float(h) * w)
            newh = short_size
    else:
        if w < h:
            neww = int(long_size / float(h) * w)
            newh = long_size
        else:
            neww = long_size
            newh = int(long_size / float(w) * h)
    image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_CUBIC)
    return image
