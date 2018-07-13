#!/usr/bin/env python

import PIL.Image
import errno
import glob
import numpy as np
import os
import skimage.measure
import sys

#input_path = sys.argv[1]
#output_path = sys.argv[2]
input_path = "./test"
ref_path = os.path.join(input_path, 'ref')
res_path = os.path.join(input_path, 'Results')
output_path = res_path

try:
    os.makedirs(output_path)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

mse = []
psnr = []
ssim = []

for fp in glob.glob(os.path.join(ref_path, '*.jpg')):
    try:
        mse.append(np.inf)
        psnr.append(0)
        ssim.append(0)

        y = np.array(PIL.Image.open(fp).convert('L')) / 255.0
        y_hat = np.array(PIL.Image.open(os.path.join(res_path, os.path.split(fp)[1])).convert('L').resize((275, 400))) / 255.0
        mse[-1] = skimage.measure.compare_mse(y, y_hat)
        psnr[-1] = skimage.measure.compare_psnr(y, y_hat)
        ssim[-1] = skimage.measure.compare_ssim(y, y_hat)
    except:
        pass

with open(os.path.join(output_path, 'scores.txt'), 'w') as f:
    f.write('mse: {}\npsnr: {}\nssim: {}'.format(np.mean(mse), np.mean(psnr), np.mean(ssim)))
