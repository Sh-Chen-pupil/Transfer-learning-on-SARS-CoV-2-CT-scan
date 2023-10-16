# Transfer-learning-on-SARS-CoV-2-CT-scan
Used three famous deep learning model(MoblieNetV2, ResNet50, VGG19) to analyze CV image with the application of  dynamic learning rate  
### Background Introduction
Since the outbreak of the COVID-19 pandemic, it has rapidly spread worldwide. Variants of the virus have enhanced its transmission capabilities, leading to a rapid increase in the number of infections in Europe and the United States. Some regions in China have also experienced multiple waves of outbreaks. COVID-19 continues to disrupt people's normal lives and poses a threat to their health. This case uses the SARS-CoV-2 CT-scan public dataset (https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset) to classify CT images using a transfer learning model (Inception V3). The goal is to efficiently differentiate between CT images of individuals infected with COVID-19 and those who are not.
### Process
(1) Perform data augmentation while reading images.

(2) Choose the network for building a transfer learning model (MobileNetV2, ResNet50, VGG19).

(3) Configure the layers at the end of the network and modify certain hyperparameters.

(4) Evaluate the three models using ROC curves (MobileNetV2 > VGG19 > ResNet50).

(5) Retrain the models using a dynamic learning rate setting for the hyperparameter.

(6) Overall, the results are better than the initial three models.

(7) Evaluate the models using other metrics (sensitivity, specificity, PPV, NPV).

(8) Visualize some of the predicted results.

Results: The model results show that the transfer learning model based on MobileNetV2 (combined with dynamic learning rate) performs the best, with an accuracy of 0.9012 and an AUC of 0.96.

