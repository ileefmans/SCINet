# SCINet


![Insight Program](https://img.shields.io/badge/Insight-Artificial%20Intelligence-lightgrey)
![GitHub last commit](https://img.shields.io/github/last-commit/ileefmans/SCINet)
![GitHub repo size](https://img.shields.io/github/repo-size/ileefmans/SCINet.svg)
![GitHub top language](https://img.shields.io/github/languages/top/ileefmans/SCINet)

## Motivation
According to the American Academy of Dermatologist Association, **50 million** people a year experience some form of acne while only **5 million** people a year actually seek professional treatment. This gap between  the number of people affected and the number of people seeking treatment coupled with a global pandemic has resulted in a huge demand for remote treatment options. When providing dermatological treatment, identifying persistent skin conditions is essential because this information could mean the difference between continuing or altering a treatment plan. It was under this context that *SCINet* was born.  

I consulted for Cureskin, a company that provides remote dermatological treatment. I was tasked with delivering a codebase for a model that helped automate the process of identifying persistent skin conditions for Cureskin.

## Overview
*SCINet* identifies facial skin conditions which are persistent over time. Given two facial images of the same patient at different stages in their treatment, and with their conditions already detected, *SCINet* identifies the corresponding conditions in each image. These corresponding conditions are considered by Cureskin to be persistent conditions. 
  
  The link to below leads to a quick demo providing a high-level intuition about what this module does.


[<img src="https://github.com/ileefmans/Re-Identifying_Persistent_Skin_Conditions/blob/master/images/Screen_Shot_2020-10-06.png" width=300 align=center>](https://www.youtube.com/watch?v=fg9VBqtjan4)  

SCINet leverages opencv's pretrained facial detection model to align the faces and then match corresponding skin conditions
  
  
  # Getting Started  
  Follow the instructions below to get clone the repo and leverage the model.
  ## Usage  
  
  *Note that model weights the data these models were trained and evaluated on are not available on this repo as they are the property of Cureskin*
  
  <font size="0.5"> *This is my text number1*</font>
  
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
  
  
  
  
  
