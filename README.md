# simple_object_localization_app

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project is to localization and predict an object in the image **note: this project only detect cucumber, eggplant, and mushroom due the dataset that I used only contains those object**. I also using flask as a backend to create an API and html as an interface to make a web from it.

# Dataset

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You can get the dataset from [Kaggle - Image Localization Dataset](https://www.kaggle.com/datasets/mbkinaci/image-localization-dataset), The dataset contains object image with jpg format and xml file is contains annotation from the corresponding images. 

![image](https://user-images.githubusercontent.com/91602612/215395129-8cdb0cc4-7df1-49df-9925-587cce783edc.png)

# Notebook

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I built the model in .ipynb file, I used google colab to helped me built the model and this is the explanation about the .ipynb file:
1. I test to plot image with the bounding box, I done this using ```xml.etree.ElementTree``` library to extract xml fit corresponding image, I extract  xmin, ymin, xmax, and ymax from xml file and plot the bounding box around the image using ```cv2.rectangle()``` with xmin, ymin, xmax, and ymax from the xml files, and this is the result

![image](https://user-images.githubusercontent.com/91602612/215397616-6f14fd0d-ed89-4878-b91e-f40e4ac3a818.png)

2. Then I read all xml files to extract label, xmin, ymin, xmax, and ymax from those xml files and append them into list. I encode the categorical value into numerical value **{"cucumber": 0, "eggplant": 1, "mushroom": 2}**, I also read all image files and append the image into list
3. I used ```np.array()``` to convert the lists of image files and outputs (contains label, xmin, ymin, xmax, and ymax)
4. Then I split inputs and outputs array into x_train, x_test, y_train, and y_test, using ```sklearn.model_selection.train_test_split()``` with parameters as follows **test_size = 0.3 and random_state = 42)**
5. Because y_train and y_test has 5 values contains (label, xmin, ymin, xmax, and ymax) I seperate label with other values (coordinate xmin, ymin, xmax, and ymax to build the bounding box) because our model will have 2 outputs (labels and bounding box coordinate) and 1 input (image array)
