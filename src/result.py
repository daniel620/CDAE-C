import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, save_model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D,MaxPool2D ,UpSampling2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
# adam = adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import cv2
from math import log10,sqrt
from PIL import Image
tf.keras.metrics.Recall()
tf.keras.metrics.AUC()
tf.keras.metrics.FalsePositives()
tf.keras.metrics.FalseNegatives()
tf.keras.metrics.

CDAE = load_model('weights/CDAE_model_1.h5')
encoder = load_model('weights/encoder_model_1.h5')
classifier = load_model('weights/Classifier_model_1.h5')

# test_noisy_img_path = '/Users/kuzaowuwei/Documents/GitHub/CDAE-C/data/preprocess/3-noisy_bmp/1.3.6.1.4.1.14519.5.2.1.6279.6001.227962600322799211676960828223/135.bmp'
# test_raw_img_path = '/Users/kuzaowuwei/Documents/GitHub/CDAE-C/data/preprocess/1-raw_bmp/1.3.6.1.4.1.14519.5.2.1.6279.6001.227962600322799211676960828223/135.bmp'
# # noisy_5e4_path = '/Users/kuzaowuwei/Documents/GitHub/CDAE-C/data/preprocess/3-noisy_bmp/1.3.6.1.4.1.14519.5.2.1.6279.6001.227962600322799211676960828223/135(5e4).bmp'

# size = 256
# test_noisy_img = np.array(Image.open(test_noisy_img_path).resize((size,size),Image.LANCZOS))/ 255
# test_noisy_img = np.expand_dims(test_noisy_img,axis=0)
# test_raw_img = np.array(Image.open(test_raw_img_path).resize((size,size),Image.LANCZOS))/ 255
# test_raw_img = np.expand_dims(test_raw_img,axis=0)

# pred_denoied_img = CDAE.predict(test_noisy_img)
# f,ax=plt.subplots(1,3)
# f.set_size_inches(20,10)
# ax[0].imshow(np.squeeze(test_raw_img) , cmap='gray')
# ax[1].imshow(np.squeeze(test_noisy_img) , cmap='gray')
# ax[2].imshow(np.squeeze(pred_denoied_img) , cmap='gray')
# # plt.axis('off')
# plt.savefig('output/CDAE_result.svg' , bbox_inches='tight')
# plt.show()


# CT result
test_noisy_img_path = '/Users/kuzaowuwei/Documents/GitHub/CDAE-C/data/preprocess/3-noisy_bmp/1.3.6.1.4.1.14519.5.2.1.6279.6001.227962600322799211676960828223/135.bmp'
test_raw_img_path = '/Users/kuzaowuwei/Documents/GitHub/CDAE-C/data/preprocess/1-raw_bmp/1.3.6.1.4.1.14519.5.2.1.6279.6001.227962600322799211676960828223/135.bmp'

size = 256
test_noisy_img = np.array(Image.open(test_noisy_img_path).resize((size,size),Image.LANCZOS))/ 255
test_noisy_img = np.expand_dims(test_noisy_img,axis=0)
test_raw_img = np.array(Image.open(test_raw_img_path).resize((size,size),Image.LANCZOS))/ 255
test_raw_img = np.expand_dims(test_raw_img,axis=0)
pred_denoied_img = CDAE.predict(test_noisy_img)

# plt.imshow(np.squeeze(test_raw_img) , cmap='gray')
# plt.axis('off')
# plt.savefig('output/CDAE_result_raw.png',dpi=500 , bbox_inches='tight')
# plt.show()
# plt.imshow(np.squeeze(test_noisy_img) , cmap='gray')
# plt.axis('off')
# plt.savefig('output/CDAE_result_noisy.png',dpi=500 , bbox_inches='tight')
# plt.show()
# plt.imshow(np.squeeze(pred_denoied_img) , cmap='gray')
# plt.axis('off')
# plt.savefig('output/CDAE_result_denoised.png' ,dpi=500, bbox_inches='tight')
# plt.show()

# Zoomed

# plt.imshow(np.squeeze(test_raw_img)[128:128+64,32:32+64] , cmap='gray')
# plt.axis('off')
# plt.savefig('output/CDAE_result_raw_zoomed.png',dpi=500 , bbox_inches='tight')
# plt.show()
# plt.imshow(np.squeeze(test_noisy_img)[128:128+64,32:32+64] , cmap='gray')
# plt.axis('off')
# plt.savefig('output/CDAE_result_noisy_zoomed.png',dpi=500 , bbox_inches='tight')
# plt.show()
# plt.imshow(np.squeeze(pred_denoied_img)[128:128+64,32:32+64] , cmap='gray')
# plt.axis('off')
# plt.savefig('output/CDAE_result_denoised_zoomed.png' ,dpi=500, bbox_inches='tight')
# plt.show()

# Zoomed 1
plt.imshow(np.squeeze(test_raw_img)[128-32:128+64,32:32+96] , cmap='gray')
# plt.annotate('',xy=(20, 85),xytext=(45,85) ,arrowprops=dict(arrowstyle="->", color="r", hatch='*',lw=3,head_length=0.4))
plt.annotate('',xy=(30, 75),xytext=(55,75) ,arrowprops=dict(color="r", hatch='*',lw=3,frac=0.4))
plt.annotate('',xy=(35, 40),xytext=(35,15) ,arrowprops=dict(color="c", hatch='*',lw=3,frac=0.4))
plt.axis('off')
plt.savefig('output/CDAE_result_raw_zoomed_96.png',dpi=500 , bbox_inches='tight')
plt.show()
plt.imshow(np.squeeze(test_noisy_img)[128-32:128+64,32:32+96] , cmap='gray')
plt.annotate('',xy=(30, 75),xytext=(55,75) ,arrowprops=dict(color="r", hatch='*',lw=3,frac=0.4))
plt.annotate('',xy=(35, 40),xytext=(35,15) ,arrowprops=dict(color="c", hatch='*',lw=3,frac=0.4))
plt.axis('off')
plt.savefig('output/CDAE_result_noisy_zoomed_96.png',dpi=500 , bbox_inches='tight')
plt.show()
plt.imshow(np.squeeze(pred_denoied_img)[128-32:128+64,32:32+96] , cmap='gray')
plt.annotate('',xy=(30, 75),xytext=(55,75) ,arrowprops=dict(color="r", hatch='*',lw=3,frac=0.4))
plt.annotate('',xy=(35, 40),xytext=(35,15) ,arrowprops=dict(color="c", hatch='*',lw=3,frac=0.4))
plt.axis('off')
plt.savefig('output/CDAE_result_denoised_zoomed_96.png' ,dpi=500, bbox_inches='tight')
plt.show()



# test_img_path = '/Users/kuzaowuwei/Documents/GitHub/CDAE-C/data/preprocess/5-ROI_classify/1/1/2.bmp'
# size = 128
# test_img = np.array(Image.open(test_img_path).resize((size,size),Image.LANCZOS))/ 255
# test_img = np.expand_dims(test_img,axis=0)
# pred_img = CDAE.predict(test_img)
# f,ax=plt.subplots(1,2)
# f.set_size_inches(40,20)
# ax[0].imshow(test_img[0] , cmap='gray')
# ax[1].imshow(np.squeeze(pred_img[0]) , cmap='gray')
# plt.savefig('output/CDAE_node_result_128.svg' , bbox_inches='tight')
# plt.show()

# test_img_path = '/Users/kuzaowuwei/Documents/GitHub/CDAE-C/data/preprocess/5-ROI_classify/1/1/2.bmp'
# size = 48
# test_img = np.array(Image.open(test_img_path).resize((size,size),Image.LANCZOS))/ 255
# test_img = np.expand_dims(test_img,axis=0)
# pred_img = CDAE.predict(test_img)
# f,ax=plt.subplots(1,2)
# f.set_size_inches(40,20)
# ax[0].imshow(test_img[0] , cmap='gray')
# ax[1].imshow(np.squeeze(pred_img[0]) , cmap='gray')
# plt.savefig('output/CDAE_node_result_48.svg' , bbox_inches='tight')
# plt.show()