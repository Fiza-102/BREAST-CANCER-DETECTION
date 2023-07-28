# BREAST-CANCER-DETECTION


## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installations](#installations)
- [Results](#results)
## Overview
This analysis aims to observe which features are most helpful in predicting malignant (cancerous) or benign (non-cancerous) and to see general trends that may aid us in model selection and hyper parameter selection. The goal is to classify whether the breast cancer is benign or malignant. To achieve this i have used machine learning classification methods to fit a function that can predict the discrete class of new input.

## Dataset
The dataset is collected from kaggle which has following Attribute Information:

- ID number 
- Diagnosis (M = malignant, B = benign) 3–32)
<br>Ten real-valued features are computed for each cell nucleus:
- radius (mean of distances from center to points on the perimeter)
- texture (standard deviation of gray-scale values)
-perimeter
-area
-smoothness (local variation in radius lengths)
-compactness (perimeter² / area — 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
-symmetry
- fractal dimension (“coastline approximation” — 1)
In our dataset we have the outcome variable or Dependent variable i.e Y having only two set of values, either M (Malign) or B(Benign). So we will use Classification algorithm of supervised learning.
## Prerequisites
 jupyter notebook, python libraries: Numpy, Pandas, seaborn, Label encoder, KNN, Logistic Regression.
## Installations
```r
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
```
## Results

In this project in python, I've build a breast cancer  tumour  predictor  on the dataset and created  graphs and results for the same. It has been observed that  a  good dataset  provides better accuracy. Selection of appropriate  algorithms with good home dataset will lead to the development of prediction  systems.  These  systems  can  assist  in proper  treatment  methods  for a  patient  diagnosed with breast cancer. I've used logistic regression model and KNN model and out of both KNN performs better.
