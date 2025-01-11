import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

# Modeli yükle (MobilNetV2 modelini kullanıyoruz)
model = tf.saved_model.load("ssd_mobilenet_v2_coco/saved_model")

# Nesne algılama fonksiyonu
def nesne_tespit_et(image_path):
    # Görüntüyü yükle
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (300, 300))



# Görüntü yolunu girin
image_path = "your_image_path_here.jpg"
nesne_tespit_et(image_path)
