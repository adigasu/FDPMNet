Fingerprint denoising M-net using Keras.

Dependencies
This code depends on the following libraries:
Keras>=2.0
theano or tenserflow

Also, this code should be compatible with Python versions 2.7-3.5. (tested in python2.7)

Example
if Cuda enabled
$ CUDA_VISIBLE_DEVICES=0 python2.7 test.py "test_path"

else

$ python2.7 test.py "test_path"

You will see the predicted results of test image in "test_path/Results"
