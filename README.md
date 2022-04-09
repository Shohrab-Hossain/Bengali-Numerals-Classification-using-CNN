# Bengali Numerals Recognition using CNN


<p align=center>
    <img src="readme-lib\Logo.png" alt="Logo" width="80%" style="min-width:150px;" />
</p>


**Digital image processing** investigates the portrayal and the control of *pictorial data*. With the development in *Machine Learning*, digital image processing is being promised over time. This research is an example of **Optical Character Recognition (OCR)**, where `Convolutional Neural Network (CNN)` is used for the *classification* of **`Bengali Numerals`**.

</br>

> Topics of this study are listed below:

1. [Image Processing](#image-processing)
     * [Optical Character Recognition (OCR)](#optical-character-recognition-(ocr))
2. [Introduction to Research Planning](#introduction-to-research-planning)
   - [Methodology](#methodology)
   - [Design of a Convolutional Neural Network](#design-of-a-convolutional-neural-network)
3. [Implementation in MATLAB](#implementation-in-matlab)
4. [Result Analysis](#result-analysis)
5. [Conclusion](#conclusion)


</br>

# Image Processing

**Image processing** is the technology that processes *a digital image* analysing its features, and then predicts information about the image. Image processing uses different *machine learning tools* to classify images based on the available features. 

The machine learning tools and algorithm consists of `Support Vector Machine (SVM)`, `k-nearest neighbour`, `Artificial Neural Network (ANN)`, etc. *Deep learning* is efficient for image processing among all other tools. Here are many deep learning models: `Deep Belief Network (DBN)`, `Restricted Boltzmann Machine (RBM)`, `Convolutional Neural Network (CNN)`, etc. 

The **Convolutional Neural Network (CNN)** offers unique features that make this model more suitable for image processing. One of the significant features of CCN is *eliminating the need* for `manual feature extraction`. Another essential feature is that one CNN model can be used for *different recognition tasks*. The existing CNN model can learn to recognise new patterns with new learning or training data.


</br>

## Optical Character Recognition (OCR)

*Image processing* can recognise `patterns` in an image. First, the system is trained with a vast set of images, from where the system learns to *classify* and *identify* patterns. Then, if a test mage is provided, the system can recognise patterns the image contains. Similarly, a system can recognise characters in an image considering the characters as a pattern. This approach is considered **Optical Character Recognition (OCR)**.



</br>
</br>

# Introduction to Research Planning

This research is based on **Optical Character Recognition (OCR)**. In this study, a system will be designed to recognise `Bengali numerals`. The numerals are *handwritten*, and the numerical dataset will be collected from a cloud-based repository. Data will be preprocessed first and then randomly split into training and testing data. Here a **`Convolutional Neural Network (CNN)`** will be designed. The CNN will be trained using the training dataset of Bengali handwritten numerical. After training, the CNN will be tested using the testing dataset for recognition accuracy.

</br>

## Methodology

This study aims to deploy a CNN for Optical Character Recognition. The methodology of this research is shown in this figure.

</br>
</br>

<p align=center>
    <img src="readme-lib\Methodology.png" alt="Methodology" width="40%" style="min-width:150px;" />
</p>
  

</br>
</br>

## Design of a Convolutional Neural Network

The CNN also consists of input and output layers like other neural networks. In between these two layers, there are many hidden layers. Some hidden layers are for feature extraction and a layer for classification.

</br>
</br>

<p align=center>
    <img src="readme-lib\Design of CNN.png" alt="Design of CNN" width="100%" style="min-width:150px;" />
</p>


</br>
</br>

Details about each of these layers can be found [here](https://uk.mathworks.com/discovery/convolutional-neural-network-matlab.html).


</br>
</br>


# Implementation in MATLAB

This research is implemented in MATLAB, and the version used is R2021b. MATLAB has excellent resources for deep neural networks, and also it provides inbuilt functions for different NN applications.

</br>

## Data Collection

For this study, a dataset of handwritten Bengali numerals is required. The dataset is collected from a [cloud-based repository](https://data.mendeley.com/datasets/hf6sf8zrkc/2?__cf_chl_captcha_tk__=0eef3f155bb8cdbd4318462de9ce43b06e8b36fe-1576926645-0-ARwIJOlc0pqWADHpP7QJNSnlLmWPdU95TWcw8RABhCTk-MHuKadBfXUxQednw2omNlWNN2YFffaQjm6HFtzYhbKIaX6ZXIydnz6nMJUQL1p3MoCjLaD8C1Y7P2F33-MAWE3Eo1NstIjhrEzpMBoz4nmc0z8Fx1uftg5rDt5YCNCiWsBY4VcaijT_NXaReFueYu3CBgk-he-9c_Xr7FhsPoQCpg8ThfHCmELrGY8f5EKmi5eFhBEWP6-1JpByjl4Z4Qr5-3zWmjn0uBsoPF2DKD6SE10QnF1hcUP5rHbSfV4Iv8mLUYoTLLeArXclMmeNEQuWKo_AgS5fAzLFuFCf_DI). The images of 10 Bengali numerals from this dataset are collected for this study. The collected dataset can be found in this GitHub repository.

</br>

**Table 01: Number of images in each class**

| **Class** | **Number of Images** | Class | **Number of Images** |
| :-------: | :------------------: | :---: | :------------------: |
|     0     |         1982         |   5   |         1986         |
|     1     |         1982         |   6   |         1981         |
|     2     |         1953         |   7   |         1958         |
|     3     |         1975         |   8   |         1984         |
|     4     |         1980         |   9   |         1967         |


</br>

## Data Preprocessing

The collected data are **preprocessed** to meet the *criteria* for the CNN. The designed CNN model requires images with dimensions `150-by-150` and have a `grey-scale` colour channel. The collected images have a grey-scale channel, but they don't match the dimension requirement. So, the images are *resized* to make the dimension150-by-150.


</br>


## Splitting Data into Training and Testing Data

After preprocessing, data are split into training data and testing data. **`75%`** of data are used for *training*, and **`25%`** are for *validation*.


</br>

## Creation of CNN

The CNN model has **four layers**. An `input layer`, `convolution layer`, `classification layer`, and `output`.

`Input layer` parameter is **`[150 150 1]`**, which means the image dimensions are 150-by-150, and one means it will process grey-scale image.

The designed CNN has four `convolution layers` for feature extraction. Each layer has a different filter size, and also the number of filters is increased in each layer. A max-pooling layer between each layer will perform the down-sampling operation.

There is a `fully connected classification layer` with ten classes, as Bengali numerals have ten digits. The SoftMax layer and classification layer follow this layer.

The classification layer generates prediction values for each of the classes. The class with the maximum prediction will be considered the output for the image that is being tested.


</br>

## Training the Network

The training process starts with a **`0.001`** learning rate. The max epoch will be ten during training, and the data will be shuffled after each epoch. The network is trained using the training data, **`75%`** of the total data. 

</br>

## Testing the Network

While training the network, the software calculates the accuracy of the testing data at regular intervals. At the end of the training, the software calculates the total accuracy.



</br>
</br>

# Result Analysis

The network is trained with images having *150-pixels* height, *150-pixels* width, and a bit depth of *8-bit*. The training images of a grey-scale channel. The machine used for training has `8.00 Gigabyte` RAM, a processor `Intel(R) Core(TM) i7-4500UCPU` with speed `1.80GHz` and `2.40 GHz`, and a `64-bit` Operating System.


</br>

> The detail about training progress is shown in this figure.


<p align=center>
    <img src="readme-lib\Training Progress of CNN.png" alt="Training Progress of CNN" width="100%" style="min-width:150px;" />
</p>


</br>


The network training is completed in **`10 epochs`**, and the total training time is `107 minutes and 25 seconds`. The network has a learning rate `of 0.001`.
When training is complete, the Convolutional Neural Network achieve an accuracy of **95.05%**. The recognition accuracy of some randomly selected Bengali numerals from testing data is shown in this figure.

</br>

<p align=center>
    <img src="readme-lib\validation.jpg" alt="validation" width="70%" style="min-width:150px;" />
</p>


</br>
</br>


# Conclusion

In this research, a **`Convolutional Neural Network`** is designed for **Optical Character Recognition** of **Bengali numerals**. The designed CNN is *fonts* and *shapes* autonomous, meaning it can detect digits of any font style, size and shape. As CNN is used, this study does not require manual feature extraction. The designed CNN extracts the features automatically, for that it requires a massive set of data during training. The CNN is trained using around 2000 images per class. The network can process grey-scale images with dimensions 150-by-150 and a bit depth of 8bit, meaning the value of each pixel is in the range of 0 to 255.



</br>
</br>



---

</br>



# Important Links

1. Documentation of Convolution Neural Network :  [Link](https://uk.mathworks.com/discovery/convolutional-neural-network-matlab.html)
2. Cloud repository of Bengali Dataset : [Link](https://data.mendeley.com/datasets/hf6sf8zrkc/2?__cf_chl_captcha_tk__=0eef3f155bb8cdbd4318462de9ce43b06e8b36fe-1576926645-0-ARwIJOlc0pqWADHpP7QJNSnlLmWPdU95TWcw8RABhCTk-MHuKadBfXUxQednw2omNlWNN2YFffaQjm6HFtzYhbKIaX6ZXIydnz6nMJUQL1p3MoCjLaD8C1Y7P2F33-MAWE3Eo1NstIjhrEzpMBoz4nmc0z8Fx1uftg5rDt5YCNCiWsBY4VcaijT_NXaReFueYu3CBgk-he-9c_Xr7FhsPoQCpg8ThfHCmELrGY8f5EKmi5eFhBEWP6-1JpByjl4Z4Qr5-3zWmjn0uBsoPF2DKD6SE10QnF1hcUP5rHbSfV4Iv8mLUYoTLLeArXclMmeNEQuWKo_AgS5fAzLFuFCf_DI) 


</br>
</br>
