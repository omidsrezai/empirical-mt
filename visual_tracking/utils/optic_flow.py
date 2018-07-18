import cv2
from skimage import img_as_ubyte, color
import numpy as np

import pyflow


def compute_optic_flow(frame1, frame2, method='fb', **kwargs):
    """Compute optic flow

    :param frame1: first rgb frame in float32
    :param frame2: second rgb frame in float32
    :param method: optic flow method
    :return: [u, v] where u is horizontal flow, v is vertical flow
    """

    methods = {
        'fb': _comput_flow_farnback,
        'c2f': _flow_coarse2fine
    }
    return methods[method](frame1, frame2, **kwargs)


def _comput_flow_farnback(frame1, frame2, **kwargs):
    f1_u8_gray = img_as_ubyte(color.rgb2gray(frame1))
    f2_u8_gray = img_as_ubyte(color.rgb2gray(frame2))

    flow = cv2.calcOpticalFlowFarneback(f1_u8_gray,
                                        f2_u8_gray,
                                        flow=None,
                                        pyr_scale=0.5,
                                        levels=4,
                                        winsize=15,
                                        iterations=3,
                                        poly_n=5,
                                        poly_sigma=1.1,
                                        flags=0)
    return flow


def _flow_coarse2fine(frame1, frame2, **kwargs):
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # RGB

    frame1_float64 = frame1.astype(np.double)
    frame2_float64 = frame2.astype(np.double)

    u, v, _ = pyflow.coarse2fine_flow(frame1_float64, frame2_float64,
                                      alpha, ratio, minWidth,
                                      nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)

    flow = np.stack([u, v], axis=2).astype(np.float32)

    return flow