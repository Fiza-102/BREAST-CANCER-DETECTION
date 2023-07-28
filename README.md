# BREAST-CANCER-DETECTION


## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installations](#installations)
- [Results](#results)
## Overview
This analysis aims to observe which features are most helpful in predicting malignant or benign cancer and to see general trends that may aid us in model selection and hyper parameter selection. The goal is to classify whether the breast cancer is benign or malignant. To achieve this i have used machine learning classification methods to fit a function that can predict the discrete class of new input.
## Dataset
The dataset is collected from kaggle which has following Attribute Information:

- ID number 
- Diagnosis (M = malignant, B = benign) 3–32)
Ten real-valued features are computed for each cell nucleus:
-radius (mean of distances from center to points on the perimeter)
-texture (standard deviation of gray-scale values)
-perimeter
-area
-smoothness (local variation in radius lengths)
-compactness (perimeter² / area — 1.0)
-concavity (severity of concave portions of the contour)
-concave points (number of concave portions of the contour)
-symmetry
-fractal dimension (“coastline approximation” — 1)
## Prerequisites
 jupyter notebook, python libraries: Numpy, Pandas, seaborn, Label encoder, KNN, Logistic Regression.
## Installations
```r
import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
```
## Results
To properly format our dataset, I first pre-processed the data. After that, I trained model using this dataset.
The model was able to generate precise predictions after being trained. This model can be utilised in actual applications going forward.
