# Network Attack Outlier/Anomaly Detection

[![Pandas](https://img.shields.io/badge/pandas-latest-green)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-green)](https://scikit-learn.org/stable/)
[![Streamlit](https://img.shields.io/badge/streamlit-latest-blue)](https://www.streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-latest-blue)](https://www.docker.com/)

In this project, we use the isolation forest model to detect anomalous network connections in a dataset of physical host network data. Our goal is to identify which hosts are likely to be outliers based on their connection characteristics.

## Data Exploration
I begin by exploring the dataset to gain an understanding of the features and their distributions. Some key observations from our data exploration include:

The duration feature is highly skewed, with the majority of connections lasting only a few seconds.
The src_bytes and dst_bytes features also have a skewed distribution, with the majority of connections transferring a small amount of data.
Algorithm Selection
For this task, I choose to use the isolation forest model because it is well-suited for detecting anomalies in datasets with many features. This is because the isolation forest model works by randomly selecting a feature and a split value and using this to isolate individual observations. Anomalous observations are more likely to be isolated earlier because they are different from the majority of the observations, and this process is repeated until each observation is isolated.

## Approach
My approach to solving this problem involves the following steps:

* Preprocessing the data by removing any null values and scaling the features.
* Training the isolation forest model on the preprocessed data.
* Using the trained model to make predictions on the test set.
* Evaluating the model's performance using metrics such as accuracy and recall, and visualizing the results with a confusion matrix.

## Evaluation
I evaluate the performance of my model using a variety of metrics, including accuracy and recall. The confusion matrix allows us to visualize the model's predictions and see how well it is able to distinguish between normal and anomalous connections.

Based on these evaluations, I find that my model performs well, with an accuracy of 0.9998207815482916:
<img src="https://i.ibb.co/hmyfTfG/Screen-Shot-2023-01-02-at-3-19-02.png" />

## Docker Application
In addition to the model and analysis, I have also developed a Docker application with a user interface that allows users to input new incidents and get a prediction of whether or not they are anomalous. The application can be installed from my GitHub project.
<img src="https://i.ibb.co/60pP4V0/Screen-Shot-2023-01-02-at-3-19-20.png" />

## Conclusion
Overall, I have successfully used the isolation forest model to detect anomalous network connections in this dataset of physical host network data. My approach has achieved good results in terms of accuracy and recall, and the Docker application provides a useful tool for predicting anomalies in real-time.
