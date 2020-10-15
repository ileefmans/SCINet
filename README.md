![Insight Program](https://img.shields.io/badge/Insight-Artificial%20Intelligence-lightgrey)
![GitHub last commit](https://img.shields.io/github/last-commit/ileefmans/SCINet)
![GitHub repo size](https://img.shields.io/github/repo-size/ileefmans/SCINet.svg)
![GitHub top language](https://img.shields.io/github/languages/top/ileefmans/SCINet)

# SCINet

**Table of Contents**  
- [Motivation](https://github.com/ileefmans/SCINet/blob/master/README.md#motivation)  
- [Overview](https://github.com/ileefmans/SCINet#overview)    
  - [*SCINet1.0*](https://github.com/ileefmans/SCINet#scinet10)     
  - [*SCINet2.0*](https://github.com/ileefmans/SCINet#scinet20)    
- [Getting Started](https://github.com/ileefmans/SCINet#getting-started)  
  - [Usage](https://github.com/ileefmans/SCINet#usage) 
- [Acknowledgments and Contributions](https://github.com/ileefmans/SCINet#acknowledgments-and-contributions)  

## Motivation
According to the American Academy of Dermatologist Association, **50 million** people a year experience some form of acne while only **5 million** people a year actually seek professional treatment. This gap between  the number of people affected and the number of people seeking treatment coupled with a global pandemic has resulted in a huge demand for remote treatment options. When providing dermatological treatment, identifying persistent skin conditions is essential because this information could mean the difference between continuing or altering a treatment plan. It was under this context that *SCINet* was born.  

I consulted for Cureskin, a company that provides remote dermatological treatment. I was tasked with delivering a codebase for a model that helped automate the process of identifying persistent skin conditions for Cureskin.

## Overview
*SCINet* identifies facial skin conditions which are persistent over time. Given two facial images of the same patient at different stages in their treatment, and with their conditions already detected, *SCINet* identifies the corresponding conditions in each image. These corresponding conditions are considered by Cureskin to be persistent conditions. 

<img src="https://github.com/ileefmans/SCINet/blob/master/images/example_image.png" width=700 align=center>
  
  This [link](https://www.youtube.com/watch?v=fg9VBqtjan4) also provides a quick demo providing a high-level intuition about what this project does. In the next two sections I will quickly detail the two versions of *SCINet* and how they differ.
  
### *SCINet1.0*
*SCINet1.0* leverages classical computer vision techniques to match the conditions in the two images. This model detects the face in each image, aligns and centers the faces using facial landmarks, then stacks the images in order to match the conditions which occur in each image. *SCINet1.0* leverages pretrained models from ```dlib ``` and ```opencv``` such as a pretrained Convolutional Neural Net for facial detection. 

### *SCINet2.0*
*SCINet2.0* is a custom deep learning articheture built from scratch using ```pytorch```. The model maps the facial images to a higher dimensional latent space using convolutional layers along with a [Spatial Transformer Network (STN)](https://arxiv.org/abs/1506.02025). Then the model flattens this latent representation back down to a two dimensional image where the transformed boudning boxes from each image can be compared an matched. It should be noted that *SCINet2.0* is not production ready as there are a few kinks that are needed to be worked out so that the model trains more effectively.
  
  
  ## Getting Started  
  Follow the instructions below to get clone the repo and leverage the model.
  ### Usage  
  
  *Note: Model weights and data these models were trained and evaluated on are not available as they are the property of Cureskin*
  
  
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
  3. Demo: 🏃🏿‍♀️🏃🏿🏃🏾‍♀️🏃🏾🏃🏽‍♀️🏃🏽🏃🏻‍♀️🏃🏻🏃🏼‍♀️🏃🏼 
  
  
  ## Acknowledgments and Contributions  
  
  All data for training and testing *SCINet* was provided by Cureskin who owns the rights to the data along with the weights of the model. This project was completed during session 20C of the [Artificial Intelligence Fellowship at Insight Data Science](https://insightfellows.com/ai). This is a public repository and contribution by pull request is welcomed.
  
  
  
  
  
