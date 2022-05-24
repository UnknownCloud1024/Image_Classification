# Project Structure

```python
Image_Classification
|--data                    # Dataset and Test images
|--models
|    |--alexnet.py
|    |--googlenet.py
|    |--resnet.py
|    |--base_model.py            # Base model
|--utils                    # Configuration
|   |--data_utils.py            # Data preprocessing configuration
|   |--train_val_utils.py        # Model training configuration
|--weight                # Model weight file after training
|--train.py                # Training script
|--predict.py                # Prediction script
```

# Environment Configuration

```python
matplotlib==3.4.3
numpy==1.21.2
opencv-python==4.5.4.58
pillow==8.3.1
scipy==1.7.2
torch==1.9.1
torchvision==0.11.1
tqdm==4.62.3
```

# Dataset

**Download link:** http://download.tensorflow.org/example_images/flower_photos.tgz

```python
|--flower_photos
|    |--daisy
|    |--dandelion
|    |--roses
|    |--sunflowers
|    |--tulips
```

Each folder corresponds to a class of image files, which means that you can make
your own dataset for training according to this file structure.

# Run Code

1. Run **train.py** to train model
   
   Configuration
   
   model_name        # Select training model
   
   data_path              # Path of datase
   
   num_classes          # The number of classes contained in the dataset
   
   epochs                    # Training times

2. Run **predict.py** to test training result
   
   Configuration
   
   img_path                      # The path of the picture used for the test    
   
   model_name                # Select training model
   
   model_weight_path     # Model weight file

# Result show

![](C:\Users\Cloud\AppData\Roaming\marktext\images\2022-05-24-09-39-05-image.png)

# Reference

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

[2] Krizhevsky, A. (2014). One weird trick for parallelizing convolutional neural networks. arXiv preprint arXiv:1404.5997.

[3] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
