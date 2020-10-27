# Video Captioner
This project develops an auto-captioner for videos.   
In this project, the training and testing dataset is the corresponding "2014 Train/Val images" dataset from dataset introduction (https://cocodataset.org/#download). 0.1% of these datasets are used since the original datasets are too large for training.
**The model is based on a combination of CNN encoding to transfer images to feature maps and LSTM decoding to trasform feature maps into word vector.**  

 
The development process is:
* Data preprocessing:
  * Data loading: load data from json file
  * Data transformation: transform caption (text) to vector
* Model architecture:
  * Encoder: Resnet 101
  * Decoder with attention: we use other's code
* Model training: record loss, top-5 accuracy, bleu score for training and testing dataset
* Inference: still in progress
  * Single pictures (from testing set): calculate the accuracy compared with its original captions
  * Video caption: record auto-caption and decoding time

## 1. Data preprocessing
As the first step of this project, we have to ensure data quality and transform captions (text) into word vector (number) for training. 

### 1.1 Data loading
In data loading, we first unzip and load training and testing data to the workplace.   
As described in dataset introduction, each image has 5 captions, so we set `captions_per_image = 5` for further checking and modelling.  
To standardize the length of captions for each image, we set `max_len = 50`, where we only keep those descriptions whose length is smaller or equal to `max_len`.   
Finally, we set up the `word_set` which is a set of words appeared in all captions.  

### 1.2 Data transformation
In data transformation, we use HDF5 for image management, so we create `train_images.hdf5` and `val_images.hdf5`, and for each image, we set the size to be `(3, 256, 256)`.  
  
For caption transformation:  
* We create `word_map` as a hash map from word to number, and we add `<start>, <end>, <padding>` as an indicator of the beginning, end and zero padding of caption vector.
* With each image, we randomly select (with no replacement) sentences if the captions number is larger than `captions_per_image`, and randomly select (with replacement) to compliment the lack of caption data.
* We transform each caption into vector based on `word_map`


## 2. Model Architecture
This model is a combination of encoding and decoding with attention.

### 2.1 Encoder
In encoding, we use pretrained ImageNet ResNet-101 as the encoding model for images.  
Since the default average pooling will output a 100*100 image, and if we want to customize the output image embedding, we should delete the default average pooling layer, and for the default fully connected layer, since it is used for information analysis but not extraction, we do not need that layer in our model
```python
modules = list(resnet.children())[:-2]
self.resnet = nn.Sequential(*modules)
```
and use `AdaptiveAvgPool2d` to customize the architecture:
```python
self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
``` 
Since we use pretrained Resnet, we have to avoid the parameter update in the following training, while for those in `adaptive_pool` should still be trained:
```python
for param in self.resnet.parameters():
  param.requires_grad = False
```

## 3. Model training
For training dataset, we have the following process:
* Encode images into information
* Decode information into predicted caption vectors
* Calculate loss function between predicted and original captions
* Backward propagation on decoder 
* Keep track of metrics

and for testing dataset, we follow the same process except the propagation update on parameters.  
**After training 30 epoches, both the training and testing loss decrease, with testing top-5 accuracy reaches about 98.21% and bleu-4 score about 0.6278.**