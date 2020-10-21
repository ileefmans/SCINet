![Insight Program](https://img.shields.io/badge/Insight-Artificial%20Intelligence-lightgrey&?style=for-the-badge&color=lightgrey)  
  
[![Build Status](https://travis-ci.com/ileefmans/SCINet.svg?branch=master)](https://travis-ci.com/ileefmans/SCINet&?style=plastic)
![GitHub last commit](https://img.shields.io/github/last-commit/ileefmans/SCINet)
![GitHub repo size](https://img.shields.io/github/repo-size/ileefmans/SCINet.svg)
![GitHub top language](https://img.shields.io/github/languages/top/ileefmans/SCINet)

# SCINet

**Table of Contents**  
- [Motivation](https://github.com/ileefmans/SCINet/blob/master/README.md#motivation)  
- [Overview](https://github.com/ileefmans/SCINet#overview)    
  - [SCINet1.0](https://github.com/ileefmans/SCINet#scinet10)     
  - [SCINet2.0](https://github.com/ileefmans/SCINet#scinet20)  
- [Usage](https://github.com/ileefmans/SCINet#usage)  
  - [Getting Started](https://github.com/ileefmans/SCINet#getting-started)  
  - [Run Demo](https://github.com/ileefmans/SCINet#run-demo)
- [Documentation](https://github.com/ileefmans/SCINet#documentation)
- [Acknowledgments and Contributions](https://github.com/ileefmans/SCINet#acknowledgments-and-contributions)  

## Motivation
According to the American Academy of Dermatologist Association, **50 million** people a year experience some form of acne while only **5 million** people a year actually seek professional treatment. This gap between  the number of people affected and the number of people seeking treatment, coupled with a global pandemic, has resulted in a huge demand for remote treatment options. When providing dermatological treatment, identifying persistent skin conditions is essential because this information could mean the difference between continuing or altering a treatment plan. It was under this context that *SCINet* was born.  

This repository serves as a consulting project for Cureskin, a company that provides remote dermatological treatment. I was tasked with delivering a codebase for a model that helped automate the process of identifying persistent skin conditions for Cureskin.

## Overview
*SCINet* identifies facial skin conditions which are persistent over time. Given two facial images of the same patient at different stages in their treatment, and with their conditions already detected, *SCINet* identifies the corresponding conditions in each image. These corresponding conditions are considered by Cureskin to be persistent conditions. 

<img src="https://github.com/ileefmans/SCINet/blob/master/images/example_image.png" width=700 align=center>
  
  This [link](https://www.youtube.com/watch?v=fg9VBqtjan4) also shows a quick demo providing a high-level intuition about what this project does. In the next two sections I will quickly detail the two versions of *SCINet* and how they differ.
  
### SCINet1.0
*SCINet1.0* leverages classical computer vision techniques to match the conditions in the two images. This model detects the face in each image, aligns and centers the faces using facial landmarks, then stacks the images in order to match the conditions which occur in both. The two flow charts below provide a visualization of how this process works.  
  
  **Alignment and Centering:**  
    
<img src="https://github.com/ileefmans/SCINet/blob/master/images/Flow_Chart.png" width=700 align=center>  
  
    
  **Stacking and Matching**  
    
<img src="https://github.com/ileefmans/SCINet/blob/master/images/Flow_Chart2.png" width=700 align=center>  
  
  
  
  
*SCINet1.0* leverages pretrained models from ```dlib ``` and ```opencv``` such as a pretrained Convolutional Neural Net for facial detection. 

### SCINet2.0
*SCINet2.0* is a custom deep learning articheture built from scratch using ```pytorch```. The model maps the facial images to a higher dimensional latent space using convolutional layers along with a [Spatial Transformer Network (STN)](https://arxiv.org/abs/1506.02025). The model then flattens this latent representation back down to a two dimensional image where the transformed bounding boxes from each image can be compared and matched. It should be noted that *SCINet2.0* is not production ready as there are a few kinks that need to be worked out so that the model trains more effectively.
  
  
  ## Usage  
  *SCINet* is structured specifically for use by Cureskin. Because of this, there are many intricacies in the code which are specifically designed to account for the structure of Cureskin's data. The [model architecture of *SCINet2.0*](./utils/model.py), by contrast, is a fairly typical custom deep learning architecture. For those who wish to use this portion of the code for different use-cases, just follow the instructions below to get clone the repo and leverage the model.
  ### Getting Started  
  
  **Note:** *Model weights, data, and other proprietary property are not available as they are the property of Cureskin. Because of this, some packages that are rendered irrelavent by these missing items have been commented out in the [requirements.txt](./requirements.txt) file (such as ```dlib```).*
  
  
  1. Clone the repository:  
  ```shell script
  git clone https://github.com/ileefmans/SCINet
  cd SCINet
  ```  
  2. Install dependencies using [requirements.txt](./requirements.txt):
  ```shell script
  pip install -r requirements.txt
  ```  
  or 
  ```shell script 
  pip3 install -r requirements.txt
  ```
  3. Start coding!! 🤓
  
  ### Run Demo  
  For those who would like a quick demo of *SCINet2.0*'s forward pass in action, follow the [Getting Started](https://github.com/ileefmans/SCINet#getting-started) instructions and then follow these steps:  
  
  1. Install [Docker](https://docs.docker.com/get-docker/)  
  2. Build demo image with Docker:
 ```shell script
 docker build -t demo:1.0 -f DemoDockerfile .
 ```  
 3. run demo image with Docker:
 ```shell script  
 docker run demo:1.0 --size=<choose size>  
 ```
 
 
 Here ```<choose size>``` is an integer that represents the height and width of your choosing that will be used to create a minibatch of random tensors to be passed through the model. It is recommended to choose a size below 500 for optimal run time.
  
  ## Documentation  
  Currently in the process of providing documentation leveraging ```Sphinx```. This was not a part of my deliverable to Cureskin, but rather a feature that I believe would nicely complement this repo.
  
  
  ## Acknowledgments and Contributions  
  
  All data for training and testing *SCINet* was provided by Cureskin who owns the rights to the data along with the weights of the model. This project was completed during session 20C of the [Artificial Intelligence Fellowship at Insight Data Science](https://insightfellows.com/ai). This is a public repository and contribution via pull request is welcomed. 😃
  
  
  
  
  
