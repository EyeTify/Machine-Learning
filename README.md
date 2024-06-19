# 👁️Eye Disease Detection Application👁️

## 📑 Overview

Machine Learning: Building models with pre-trained model InceptionV3 and transfer learning, then converting the h5 type models to TensorFlow. Js for deployment to API. The model achieves a validation loss of 0.6790 and validation accuracy model of 81.29%.

## 📚 Related Project Repositories
| Learning Paths      | Link                                                   | 
|-------------------------|------------------------------------------------------------------|
| ☁️ Cloud Computing | [CC Repository](https://github.com/EyeTify/Cloud-Computing)   | 
| 📱 Mobile Development|  [MD Repository](https://github.com/EyeTify/Mobile-Development)    | 



## 📲 Features

- **Disease Detection**: Identifies common eye diseases such as Cataract, Uveitis, Glaucoma, Hordeolum, Conjunctivitis, Bulging, and Crossed.
- **User-Friendly Interface**: Simple and intuitive UI for uploading images and viewing results.
- **Model Explanation**: Provides explanations for the model's predictions.
- **Data Privacy**: Ensures user data is securely handled and processed.

## 📄 Dataset
https://bit.ly/EyetifyDataset

## 🛠 Tools/IDE/Library and resources
| Category        | Name            | Description                                                                                           |
|-----------------|-----------------|-------------------------------------------------------------------------------------------------------|
| IDE             | Google Colaboratory | For training models using GPUs or TPUs.                                                              |
|                 | Kaggle          | For developing models.                                                                                |
| Tools           | Python          | For providing the functionality to handle every step from data preprocessing to model deployment.   |
| Library         | TensorFlow      | For deploying our model to ensure it is ready for use in applications on mobile devices.             |
|                 | Scikit-learn    | For developing machine learning models.                                                              |
|                 | Matplotlib      | For creating visualizations of data and model results.                                                |
|                 | NumPy           | For numerical operations and data manipulation.                                                       |
| Resources       | Kaggle          | Source labeled images of eye diseases from public datasets.                                           |
|                 | Roboflow        | A platform for annotating and managing datasets for computer vision tasks.                           |


## 🔍 Data Preprocessing
Normalize and augment the images to improve model performance.

## 🔍 Model Selection
Various architectures were tested, and the final model is based on a pre-trained InceptionV3 model with transfer learning.

## 📈 Training
The model was trained using TensorFlow and Keras with cross-validation to ensure robustness. The final model achieves a validation loss of 0.6790 and a validation accuracy of 81.29%.
![Accuration](https://github.com/ayuastari/eyetify/blob/main/ACCURACY.png)
## 🔍 Fine-Tuning
Fine-tuning involves unfreezing the top layers of the pre-trained model and re-training them with a very low learning rate. This allows the model to better adapt to the specific features of the new dataset.

## 🔍 Evaluation
The model's performance was evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

## 🪄 Converting Model to TensorFlow.js
To deploy the trained model in a web application, the model is converted from a .h5 file to TensorFlow.js format.
