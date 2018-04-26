#!/usr/bin/env python3

import os
import math
import numpy as np
from PIL import Image
import argparse

filter_shape = (8, 8)


class DCT:

    def __init__(self, shape):
        self.shape = shape
        self.N = self.shape[0] * self.shape[1]
        self.cmat = np.array(
            [self._precalc(i) for i in range(self.N)], dtype=np.float)
        self.cmat[0] = np.ones(self.N) / np.sqrt(self.N)

    def __call__(self, data):
        data = data.reshape(self.N)
        return self.cmat @ data

    def inv(self, cmat):
        invmat = np.sum(self.cmat.T * cmat, axis=1)
        invmat = np.round(invmat)
        return invmat.reshape(self.shape)

    def _precalc(self, k):
        # DCT-II
        return np.cos((1 / self.N) * np.pi * k *
                      (np.arange(self.N) + 0.5)) * np.sqrt(2 * (1 / self.N))


def compress(img, threshold):
    x = np.asarray(img, dtype=np.float)
    rows = math.floor(x.shape[0] / filter_shape[0])
    cols = math.floor(x.shape[1] / filter_shape[1])
    result = np.zeros(
        (rows * filter_shape[0], cols * filter_shape[1]), dtype=np.float)
    dct = DCT(filter_shape)

    for i in range(rows):
        for j in range(cols):
            sx = i * filter_shape[0]
            sxd = sx + filter_shape[0]
            sy = j * filter_shape[1]
            syd = sy + filter_shape[1]

            sliced = x[sx:sxd, sy:syd]

            # DCT
            cmatrix = dct(sliced)

            # Threshold
            cmatrix = cmatrix * (abs(cmatrix) > (np.max(cmatrix) * threshold))

            # iDCT
            result[sx:sxd, sy:syd] = dct.inv(cmatrix)

    # clip exceeded pixels
    result[np.where(result < 0)] = 0
    result[np.where(result > 255)] = 255
    return Image.fromarray(result.astype(np.uint8))


def main(args):
    image = Image.open(args.input).convert('L')
    image.save('./original.jpg')
    compressed = compress(image, args.threshold)
    compressed.save('./compressed.jpg')
    orig_size = os.stat('original.jpg').st_size
    comp_size = os.stat('compressed.jpg').st_size
    print('threshold', args.threshold)
    print('original.jpg ->', orig_size, 'bytes')
    print('compressed.jpg ->', comp_size, 'bytes')
    print('save up to', orig_size - comp_size, 'bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-t', '--threshold', required=True, type=float)
    args = parser.parse_args()
    main(args)