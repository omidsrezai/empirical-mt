import cv2
import numpy as np
from skimage import img_as_float, img_as_float32, img_as_ubyte

import pyimgsaliency as psal

cv2_static_fine_grained_saliency = cv2.saliency.StaticSaliencyFineGrained_create()


def compute_saliency(frame, method='static_fine_grained', **kwargs):
    """Compute Saliency segmentation map

    :param frame: float32 rgb frame
    :param method: saliency method
    :return: binary segmentation map in float32
    """
    methods = {
        'static_fine_grained_thre': _compute_saliency_static_fine_grained,
        'mb': _compute_minimum_barrier,
        'rbd': _compute_robust_background_detection
    }

    # threshold the saliency map
    saliency_map_u8 = methods[method](frame, **kwargs)
    # bin_map = cv2.threshold(saliency_map_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    bin_map = img_as_float32(saliency_map_u8)

    return bin_map


def _compute_saliency_static_fine_grained(frame, **kwargs):
    frame_u8 = img_as_ubyte(frame)
    (success, saliency_map) = cv2_static_fine_grained_saliency.computeSaliency(frame_u8)

    if (not success):
        raise ValueError('static saliency fine grained failed')

    return saliency_map


def _compute_minimum_barrier(frame, **kwargs):
    frame_u8 = img_as_ubyte(frame)
    saliency_map = psal.get_saliency_mbd(imgu8=frame_u8)
    saliency_map = saliency_map.astype(np.ubyte)

    return saliency_map

def _compute_robust_background_detection(frame, **kwargs):
    frame_u8 = img_as_ubyte(frame)
    saliency_map = psal.get_saliency_rbd(rgbimg=frame_u8)
    saliency_map = saliency_map.astype(np.ubyte)

    return saliency_map