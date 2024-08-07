# Retina blood vessel segmentation

## Pre-requisite
- Ubuntu 18.04 / Windows 7 or later
- NVIDIA Graphics card


- Install Tensorflow-Gpu version-2.0.0 and Keras version-2.3.1
```
pip install tensorflow==2.10.0
```
- Install packages from requirements.txt
```
pip install -r requirements.txt
```

### DRIVE Dataset
I used DRIVE dataset, it is consist of 40 images that 6 of them used for testset and the rest for training. images in test set are chosen manually to make sure of
same distribution with training images.

### Dataset download link for DRIVE
```
https://drive.grand-challenge.org/
```

### Dataset Pre-processing
5 diffrent way of augmentation(HorizontalFlip, VerticalFlip, GridDistortion, OpticalDistortion, CoarseDropout) has done on the train data, also techniques like patching,
croping, normalization, adaptive contrast enhansement and denosing is done.

## Training
in this code you have access to 3 diffrent Unet base (Unet, Attention Unet, Unet+densenet121 as backbone) models to train them on the dataset.

- Type this in terminal to run the train.py file
```
python train.py --crop_shape=128 --bs=4 --epoch=200 --lr=2e-3 --augment=no --model=unet
```
- There are different flags to choose from. Not all of them are mandatory.

## Evaluation on test set

- Type this in terminal to run the evaluate.py file
```
python evaluate.py --stride=3 
```
- There are different flags to choose from. Not all of them are mandatory
