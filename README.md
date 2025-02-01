# Machine Learning Projects

## Overview

This repository contains machine learning projects I've conducted during the process ML exploration. Most of the projects also contain detailed reports that discuss the rationale behind the projects, methodology used and discussion/evaluation of the achieved results with further suggestions on ways of improvement. The notebook files used are also organised is a sequential step-by-step manner making them easy to follow and understand.

All the required data for the projects are available in the links contanined within the Jupyter Notebooks or folder paths.

Every project followed the standard protocol of machine learning projects including:

1. Data Processing
2. Exploratory Data Analysis (EDA)
3. Feature Selection
4. Model Training and Evaluation
5. Discussion on Results

## Project List

### [Twitter Sentiment Analysis](./twitter-sentiment/)

This project aims to understand user sentiments in tweets by constructing and evaluating supervised machine learning models. Data preprocessing involves cleansing, tokenizing, and encoding tweets, followed by vectorization using TF-IDF and Word2Vec. Four models: Logistic Regression, Linear SVC, Random Forest, and SGD Classifier—are trained on the refined dataset. The performance of these models is evaluated using accuracy, precision, recall, and F1-score metrics. Additionally, a deep learning approach using a fine-tuned BERT model is employed, achieving an accuracy of 82%, which is 4% higher than the best-performing machine learning model, Logistic Regression with Word2Vec, which had an accuracy of 78%.

### [UNet Model for Segmentation](https://github.com/ufakz/unet-final)

Deep learning project using the UNet architecture for biomedical image segmentation and road scene detection with PyTorch. Supports both standard UNet and early fusion UNet variants for RGB and depth image pairs.

Project repository: https://github.com/ufakz/unet-final

### [Sensor Fusion](./sensor-fusion/)

Using RGB and depth images, a Fully Convolutional Network (FCN) is trained for road scene detection. Unlike traditional convolutional networks that use fully connected layers at the end, FCNs use convolutional layers throughout the network allowing them to capture spatial information for segmentation tasks. Depth images provide information about the distance of objects from the camera. Each pixel in a depth image represents the distance from the camera to the object in the scene. By using both RGB and depth images, the FCN can leverage color information and spatial depth information to improve the accuracy of the detection.

Download the dataset for this project here: [Road_Scenes_Dataset](https://mega.nz/file/2x03CIYY#Z_a2YGJEWa9Yr4Uk7Qe1LQd7UyUIttsQQ_hozChdKLE)

### [Pizza Classifier](./pizza-classifier/)

In this project, a CNN was trained to classify between images of pizza and non-pizza using the pretrained ResNet101 as a backbone. Detailed step-by-step instructions are provided on how to execute the project. In the second step of the project, a more complex CNN architecture was implemented and yielded an improved +5% gain over the initial trained model. The results are discussed at the end of the project.

Download the dataset for this project here: [Pizza_Dataset](https://drive.google.com/file/d/1LTbQx71oeOm5LRCfE6vVTscdbw9j3OBR/view?usp=sharing)

### [Human Activity Clustering](./human-activity-clustering/)

In this project, we build and evaluate unsupervised machine learning models using the UCI Human Activity
Recognition dataset. The models are first evaluated without dimensionality reduction on the dataset and subsequently, newer iterations of the models are fitted after applying dimensionality reduction. The project intricately details the dataset, preprocessing, methods used to find optimal parameters for the models, impact of dimensionality reduction on the clustering process and a comparative study between the models.

### [Bank Campaign Prediction](./bank-campaign/)

In this project, we build and evaluate two supervised machine learning models designed predict whether the customer of a bank would accept a fixed term deposit. The project intricately details the criteria employed for feature selection, data preprocessing and provides a comprehensive analysis of the performance metrics associated with the selected machine learning models. As a culminating highlight, the model compares and discusses the performance of the selected machine learning models.

### [Flight Price Prediction](./flight-price-prediction/)

This project addresses the challenge faced by users in selecting the most optimal flight from a plethora of options available on various airline websites. The abundance of flight information can overwhelm users, leading to suboptimal choices and time inefficiency. The primary objective is to streamline this process by integrating data from flight websites, conducting comprehensive data processing, and employing exploratory data analysis techniques.

## Contributing

Contributions are welcome! Feel free to fork the repository, make improvements, or submit a pull request.

Happy coding !!!
