


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
  * [mix-1.ipynb](#mix-1)
  * [mix-2.ipynb](#mix-2)
  * [mix-3.ipynb](#mix-3)
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
Contains code for different models.

#### auto_encoder
Contains logic for stacked autoencoder class along with encoder and decoder.

#### cnn_classifier
Contains cnn class along with different parameters.

## notebook
Contains notebooks with different approaches in order to find the best accuracy.

### mix-1
approach1

### mix-2
approach2

### mix-3
approach3

## output
Contains saved autoencoder output

## pickle
Contains saved models autoencoder and cnn

## utils
Contains differnet utility functions

####command_parser
Contains command line argument parsing utility

####config_parser
Contain configration file reading utility

####custom_logging
Contains custom logging utility

####helper
Contains different helper functions for training of models

####stack_notification
Can be used as a notification system, when training will finish, send a message to perticuler slack channel  
Use your slack ```token='YOUR TOKEN'``` to get notification.

###requirements 
Keeps information of all the libraries used.
###config
Keeps all the hyper parameters used in models
###main
Program starter 
###training
Contains different functionalities for training and validation



##How to use this repository
1. clone the Project
2. install libraries (`pip install -r requirements.txt`)
3. get help  (`python main.py --help`)
4. execute program (`python main.py`) if no arguments passed program will take difault arguments from config file


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