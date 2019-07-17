# Cat / dog classification with CNNs

### This is an adaptation of a tutorial on CNN's by [analyticsvidhya](https://www.analyticsvidhya.com/blog/2017/06/architecture-of-convolutional-neural-networks-simplified-demystified/)
This repo is a tool for me to confirm understanding of the mechanics of CNN's and to get practice
with setting one up using Keras.

#### Setup
Download the source data from [here](http://files.fast.ai/data/dogscats.zip)

#### Running
Run train.py. modifying the number of images per category import as the sole argument

#### Notes
The original code on Analytics Vidhya relies on the CV2 library for image resizing.
I was not able to install this library on a MacOS, and so rely on the Python Image Library
instead to rescale the images to a standard size.

I trained the CNN locally on an Apple MacBook Pro with 16 GB of RAM and 4 cores.  To limit the computational 
time, I trained the network with a total of 1500 images per category.

I was not getting good accuracy with the basic architecture, so made it a little more complex:
- 2 convolutional layers
- max pooling
- convolutional layer
- batch normalization
- max pooling
- 2 dense layers with relu activation
- final dense layer with softmax

Validation accuracy is at around 75% with a stride of 2.