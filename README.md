# Low-Light-Image-Denoising

## Description
This project focuses on implementing a UNet model to perform image denoising. The UNet model is a type of convolutional neural network (CNN) that has been widely used for image segmentation and denoising tasks due to its efficient architecture for capturing both low-level and high-level features. The average PSNR for the test set is calculated to evaluate the performance of the UNet model which came out to be 20.84. Higher PSNR values indicate better image quality and more effective noise removal.

Paper used for implementation and dataset: https://arxiv.org/pdf/2404.14248

## DATA

* Training Data:The model is trained with images from the train/low and train/high directories.

  * train/low: Contains low light images.

  * train/high: Contains corresponding well Lit images.

* Testing Data:The model processes images from the test/low directory to produce enhanced outputs.

  * test/low: Contains low light images for testing.
