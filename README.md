


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

## Objective
Create a combined network of autoencoder and classifier to classify CIFAR10 data.


## Constraints
1. Use only 50% data of classes bird, deer and truck
2. Use output of encoder part of autoencoder as input of classifier

## Approach
* Create a custom dataset which reads only 50% of bird, deer and truck, remaining classes 100%
* Create sampler which assignes weights to different classes to deal with imbalance dataset
* Use Image augmentation to inhance performance of classifire
* Use stacked convolutional autoencoder to get most important features of input image
* Use convolutional neural network to classify images
* Use encoded output of autoencoder and use as input to CNN

## Architecture
<!-- PROJECT LOGO -->
![alt logo](https://raw.githubusercontent.com/DineshDevPandey/cifar/master/utils/images/logo.png)
<br />


<!-- TABLE OF CONTENTS -->
## Directory structure

* [data](#data)
* [log](#log)
* [model](#model)
  * [auto_encoder.py](#auto_encoder)
  * [cnn_classifier.py](#cnn_classifier)
* [notebook](#notebook)
* [output](#output)
* [pickle](#pickle)
* [utils](#utils)
  * [command_parser.py](#command_parser)
  * [config_parser.py](#config_parser)
  * [custom_logging.py](#custom_logging)
  * [helper.py](#helper)
  * [stack_notification.py](#stack_notification)
* [requirements.txt](#requirements)
* [config.cfg](#config)
* [main.py](#main)
* [training.py](#training)




## data
This directory contains Cifar10 dataset. If data is not downloaded already it will be downloaded and stored here.


## log
This directory contains log file generated in each run.

## model
Contains models architecture.

#### auto_encoder
Contains logic for stacked autoencoder class along with encoder and decoder.

#### cnn_classifier
Contains cnn class along with different parameters.

## notebook
Contains notebooks with different approaches in order to find the best accuracy.

## output
Contains saved autoencoder output

## pickle
Contains saved models autoencoder and cnn

## utils
Contains differnet utility functions

#### command_parser
Contains command line argument parsing utility

#### config_parser
Contain configration file reading utility

#### custom_logging
Contains custom logging utility

#### helper
Contains different helper functions for training of models

#### stack_notification
Can be used as a notification system, when training will finish, send a message to perticuler slack channel  
Use your slack ```token='YOUR TOKEN'``` to get notification.

### requirements 
Keeps information of all the libraries used.
### config
Keeps all the hyper parameters used in models
### main
Program starter 
### training
Contains different functionalities for training and validation

## Observation
To achieve better performance in any deep learning model we need to apply series of hyper parameter optimization techniques and 
find out the best combination of them.
##### Hyper parameters to tune -
* learning rate
* Number of layers in NN
* Number of nodes in each layers
* Batch size
* Number of epochs
* decay rate etc

I have tried different approaches. Key points of different approaches along with the notebook link is listed below -

Approach 1:
1. Parameter setting: batch_size = 100, learning_rate = 0.0001, num_epochs = 60, (L2 regularization)weight_decay=0.001, no image augmentation
2. Observation : 
- CNN: accuracy: Train: 93%, Test: ~75%
- By seeing accuracy graph we can say that model is overfitting. 
3. Notebook [Mix-7](https://github.com/DineshDevPandey/cifar/tree/master/notebook/mix-7.ipynb)

Approach 2:
1. Parameter setting: batch_size = 100, learning_rate = 0.0001, num_epochs = 120, (L2 regularization)weight_decay=0.001, with image augmentation
2. Observation : 
- CNN: accuracy: Train: ~90%, Test: ~72%
- By seeing accuracy graph we can say that model is overfitting. 
3. Notebook [Mix-8](https://github.com/DineshDevPandey/cifar/tree/master/notebook/mix-8.ipynb)

Approach 3:
1. Parameter setting: batch_size = 128, learning_rate = 0.0001, num_epochs = 120, (L2 regularization)weight_decay=1e-5, with image augmentation
2. Observation : 
- CNN: accuracy: Train: ~85%, Test: ~80%
- By seeing accuracy graph we can say that model is slightly overfitting. 
3. Notebook [Mix-9](https://github.com/DineshDevPandey/cifar/tree/master/notebook/mix-9.ipynb)

Approach 4:
1. Parameter setting: batch_size = 256, learning_rate = 0.0001, num_epochs = 120, (L2 regularization)weight_decay=1e-6, with image augmentation
2. Observation : 
- CNN: accuracy: Train: ~80%, Test: ~79% for epochs = ~112
- By seeing accuracy graph we can say that model is able to generalize. 
3. Notebook [Mix-10](https://github.com/DineshDevPandey/cifar/tree/master/notebook/mix-10.ipynb)

## How to use this repository
1. clone the Project
2. install libraries (`pip install -r requirements.txt`)
3. get help  (`python main.py --help`)
4. execute program (`python main.py --options`) if no arguments passed program will take default arguments from config file
###### Option list:
* valid=1  [No training, only validate with previously trained model]
* epoch=20 [Number of epochs]
* batchSize [batch size]
* weightDecay [L2 regularization]
* learningRate [learning rate]
* Example: `python main.py --valid=0 --epoch=30 --batchSize=100 --weightDecay=0.00001 --learningRate=0.0001`


<!-- LICENSE -->
## License

Distributed under the MIT License. 

<!-- CONTACT -->
## Contact


Project Link: [https://github.com/DineshDevPandey/cifar](https://github.com/DineshDevPandey/cifar)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/badge/contributers-1-brightgreen
[contributors-url]: https://github.com/DineshDevPandey/cifar/graphs/contributors
[forks-shield]: https://img.shields.io/badge/forks-0-brightgreen
[forks-url]: https://github.com/DineshDevPandey/cifar/network/members
[stars-shield]: https://img.shields.io/badge/stars-0-brightgreen
[stars-url]: https://github.com/DineshDevPandey/cifar/stargazers
[issues-shield]: https://img.shields.io/badge/issues-0-brightgreen
[issues-url]: https://github.com/DineshDevPandey/cifar/issues
[license-shield]: https://img.shields.io/badge/license-MIT-brightgreen
[license-url]: https://github.com/DineshDevPandey/cifar/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-In-brightgreen
[linkedin-url]: https://www.linkedin.com/in/dinesh-dev-pandey-51317229/
[product-screenshot]: /utils/images/logo.png