import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    
    # encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


def load_images_from_folder(folder, target_size=(256, 256)):
    images = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if filename.startswith('.'):
            continue  # Skip hidden files like .DS_Store
        try:
            img = load_img(file_path, color_mode='grayscale', target_size=target_size)
            img = img_to_array(img)
            images.append(img)
        except UnidentifiedImageError:
            print(f"UnidentifiedImageError: Cannot identify image file {file_path}")
        except Exception as e:
            print(f"Error: {e} - for file {file_path}")
    return np.array(images)


train_low_folder = 'train/low'
train_high_folder = 'train/high'

low_images = load_images_from_folder(train_low_folder)
high_images = load_images_from_folder(train_high_folder)

low_images = low_images / 255.0
high_images = high_images / 255.0

x_train, x_val, y_train, y_val = train_test_split(low_images, high_images, test_size=0.2, random_state=42)

input_size = (256, 256, 1)
model = unet_model(input_size)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1, batch_size=8, validation_data=(x_val, y_val))


model.save('unet_denoising_model.h5')


test_low_folder = 'test/low'
prediction_folder = 'test/predicted'

os.makedirs(prediction_folder, exist_ok=True)

low_test_images = load_images_from_folder(test_low_folder)
low_test_images = low_test_images / 255.0

predictions = model.predict(low_test_images)

for i, prediction in enumerate(predictions):
    prediction = (prediction * 255).astype(np.uint8)
    prediction = Image.fromarray(prediction.squeeze(), mode='L')
    prediction.save(os.path.join(prediction_folder, f'pred_{i}.png'))


def calculate_psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

test_high_folder = 'test/high'
high_test_images = load_images_from_folder(test_high_folder)
high_test_images = high_test_images / 255.0

pred_psnr = calculate_psnr(high_test_images, predictions)
avg_psnr = np.mean(pred_psnr)

print(f'Average PSNR for the test set: {avg_psnr:.2f}')
