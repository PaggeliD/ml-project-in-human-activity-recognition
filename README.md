# ML Project in Human Activity Recognition
In this projet we built four machine learning models for predicting the physical activities of ten participants through sensors that were placed in different spots on their bodies. The activities we tried to predict are jogging, downstairs, upstairs, standing, sitting, biking and walking. The ML models we built for this classification task are SVM (Support Vector Machine), KNN (K-Nearest Neighbors), Decision Trees and MLP (Multilayer Perceptron).
## Dataset
The data we used are available publicly in the project of Shoaib et al.[1]. The data contains sensor signals that were recorded out of 10 participants while performing one of the above physical activities. Every signal is a value of a smartphone accelerometer on different body spots. The last column of the data contains the physical activity at a certain moment in time.
## Data Cleaning and Pre-Processing
The pre-processing phase of the signal includes the following steps:
1. We chose a certain spot on the body which is the right pocket values.
2. We converted the 3D signal vector of the accelerometer into a scalar signal using the Euclidean norm of the acceleration of the 3D signal.
3. The values of the acceleration norm that are over 1000 are sensor errors and we replaced them each time with the previous value of the sensor.
## Feature Extraction
Windows of 20 seconds with a step of 1 second are selected. In each window we assigned the physical activity with the majority of the values in each window. From each window we extracted the following features:
1. Mean of the values of each window
2. Standard Deviation of the values of each window
3. Skewness of the values of each window
4. Maximum value
4. Minimum value
5. DIfference of maximum and minimum values
6. Spectrum power estimation using Welch's method
In the end what's left is a feature vector column and the activity column for each window of the signal.
## Training and Test Process
We trained our model using the LOSO (Leave-One-Subject-Out) Cross Validation method. With this method, we used as training set the extracted features of the 1,...,i-1,i+1,...,n participants and as a test set the extracted features of the i participant. This procedure is being repeated 10 times. Finally adding the confusion matrices together we get a final confusion matrix where we can evaluate each classification method and compute the evaluation metrics.
## Evaluation
We performed a grid search in every classification method we used to find the optimal parameters.
### SVM
The overall accuracy was 82.73% and the optimal parameters C: 100 and \gamma: 0.0078125. The following table contains the Precision, Recall and F1-Score for each class:
| Class      | Precision | Recall | F1-Score |
| :---:      | :-------: | :----: | :------: |
| Biking     | 0.97      | 0.96   | 0.96     |
| Downstairs | 0.84      | 0.94   | 0.89     |
| Jogging    | 0.99      | 0.99   | 0.99     |
| Sitting    | 0.53      | 0.88   | 0.66     |
| Standing   | 0.67      | 0.23   | 0.34     |
| Upstairs   | 0.98      | 0.97   | 0.97     |
| Walking    | 0.93      | 0.82   | 0.87     |

| Metric       | Precision | Recall | F1-Score |
| :----:       | :-------: | :----: | :------: |
| accuracy     |           |        | 0.83     |
| macro avg    | 0.84      | 0.83   | 0.81     |
| weighted avg | 0.84      | 0.83   | 0.81     |

The confusion matrix for all participants is:

### K-Nearest Neighbors
The overall accuracy was 82.75% and the optimal parameters n_neighbors: 20. The following table contains the Precision, Recall and F1-Score for each class:
| Class      | Precision | Recall | F1-Score |
| :---:      | :-------: | :----: | :------: |
| Biking     | 0.99      | 0.97   | 0.97     |
| Downstairs | 0.84      | 0.95   | 0.89     |
| Jogging    | 0.99      | 0.99   | 0.99     |
| Sitting    | 0.56      | 0.49   | 0.52     |
| Standing   | 0.54      | 0.60   | 0.57     |
| Upstairs   | 0.92      | 0.93   | 0.93     |
| Walking    | 1.00      | 0.87   | 0.93     |

| Metric       | Precision | Recall | F1-Score |
| :----:       | :-------: | :----: | :------: |
| accuracy     |           |        | 0.83     |
| macro avg    | 0.83      | 0.83   | 0.83     |
| weighted avg | 0.83      | 0.83   | 0.83     |

The confusion matrix for all participants is:

### Decision Trees
The overall accuracy is 85.92% and the optimal parameters are max_depth: 20 and criterion: entropy. The following table contains the Precision, Recall and F1-Score for each class:
| Class      | Precision | Recall | F1-Score |
| :---:      | :-------: | :----: | :------: |
| Biking     | 0.93      | 0.91   | 0.92     |
| Downstairs | 0.91      | 0.86   | 0.89     |
| Jogging    | 0.95      | 0.99   | 0.97     |
| Sitting    | 0.67      | 0.69   | 0.68     |
| Standing   | 0.69      | 0.70   | 0.69     |
| Upstairs   | 0.95      | 0.93   | 0.94     |
| Walking    | 0.92      | 0.94   | 0.93     |

| Metric       | Precision | Recall | F1-Score |
| :----:       | :-------: | :----: | :------: |
| accuracy     |           |        | 0.86     |
| macro avg    | 0.86      | 0.86   | 0.86     |
| weighted avg | 0.86      | 0.86   | 0.86     |

The confusion matrix for all participants is:

### MLP (Multilayer Perceptron)
For this neural network model we used one hidden layer with 71 neurons and the stochastic gradient descent, with momentum, as the optimizer. The overall accuracy was 84% and the optimal parameters were momentum: 0.9 and rate: 0.2. The following table contains the Precision, Recall and F1-Score for each class:
| Class      | Precision | Recall | F1-Score |
| :---:      | :-------: | :----: | :------: |
| Biking     | 0.97      | 0.95   | 0.96     |
| Downstairs | 0.81      | 0.99   | 0.89     |
| Jogging    | 0.99      | 0.98   | 0.98     |
| Sitting    | 0.58      | 0.68   | 0.63     |
| Standing   | 0.63      | 0.52   | 0.57     |
| Upstairs   | 0.98      | 0.97   | 0.98     |
| Walking    | 0.99      | 0.78   | 0.87     |

| Metric       | Precision | Recall | F1-Score |
| :----:       | :-------: | :----: | :------: |
| accuracy     |           |        | 0.84     |
| macro avg    | 0.85      | 0.84   | 0.84     |
| weighted avg | 0.85      | 0.84   | 0.84     |

### Results
It is clear that the algorithm of every classification method we used, confuses, at a very high rate, the activities of standing and sitting. We regrouped the dataset by joining together the data of standing and sitting activities as one activity. After regrouping the data, the accuracy we achieved for every classification method was: SVM 94.89%, KNN 95.26%, Decision Trees 93.84, MLP 94.51% respectively. Furthermore, we chose SVM as the optimal estimator and as train set the right pocket values of all the participants and as test set the values of the left pocket and wrist for a random participant. Remarkably enough the algorithm was able to predict the activities using as test set the left pocket values at a rate of 98.88% accuracy. On the other hand, the algorithm confused the activities at a rate of 42.06% accuracy using as test set the wrist values.


















