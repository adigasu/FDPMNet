# FPD-M-net: Fingerprint Image Denoising and Inpainting Using M-Net Based Convolutional Neural Networks using Keras.

This is end-to-end trainable Convolutional Neural Network (CNN) based architecture for fingerprint image denoising and inpainting problem. We pose the fingerprint denoising and inpainting as a segmentation (foreground) task. Our architecture is based on the M-net which was proposed for brain segmentation. We modify this architecture and call the resulting architecture as **_FPD-M-net_**. The fingerprint images are degraded with varies distortion. Our model tackles the distortions and noise using encoder layers and restores the fingerprint images using decoder layers. Our method achieves the overall 3rd rank in the _Chalearn LAP Inpainting Competition Track 3 - Fingerprint Denoising and Inpainting, ECCV 2018_ : https://competitions.codalab.org/competitions/18426#results

### Dependencies
This code depends on the following libraries:

Keras>=2.0

theano or tensorflow

Also, this code should be compatible with Python versions 2.7-3.5. (tested in python2.7)


### Example
if Cuda enabled

> $ CUDA_VISIBLE_DEVICES=0 python2.7 test.py "test_path"

else

> $ python2.7 test.py "test_path"

The predicted results will be in "test_path/Results"
