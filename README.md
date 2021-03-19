# Surface cracks detection in pavements

Capstone project for the Springboard Machine Learning Engineering Career Track

## Introduction

Computer vision is used for surface defects inspection in multiple fields, like manufacturing and civil engineering. In this project, the problem of detecting cracks in a concrete surface has been tackled using a deep learning model.

## Data

SDNET2018 is an annotated dataset of concrete images with and without cracks from bridge decks, walls and pavements [1]. The pavements subset, which includes 2600 positive images (with crack) and 21700 negative images (without crack), has been used to train and test this model. First, the data have been divided into three sets, namely train (80%), validation (10%) and test (10%), then, only for the train subset, new images with cracks have been created and saved to balance the two classes. Here an example of 9 images generated using different augmentation techniques.

![DataAugmentation](README_images/DataAugmentation.png?raw=true)
 
More information and examples about the data augmentation are reported in notebooks/DataAugmentation.ipynb  

## Model

![Architecture](README_images/Architecture.png?raw=true)

#### Convolutional Neural Network architecture

The architecture used for this project is based on convolutional neural networks (CNN) and it is inspired by the many architectures reported in literature, especially VGG16 [2]. It consists of the sequence of 5 CNN blocks, the first three blocks have a convolutional layer, followed by a batch normalization, relu activation function and a max pool layer. The last two blocks have two consecutives convolutional+batch normalization+ReLu layers before the max pool. This approach allows to increase the effective receptive field, limiting the number of trainable parameters and accelerating the training [2].
The stack of convolutional layers is followed by two fully connected layers, with a final softmax activation function that performs the binary classification.

L2 and dropout regularization have been used in the first fully connected layer to limit the overfitting. Dropout has been used only after the last batch normalization layer to avoid variance shift [3-4]. Experimentation showed that adding L2 regularization to the convolutional layers does not improve the performances.

#### Hyperparameters Tuning

Hyperparameters tuning is critical to optimize the performance of a neural network and is very time consuming, since there are infinite combinations of hyperparameters that can be tested. In this section, the impact of few hyperparameters is summarized, namely learning rate, padding and regularization.

##### Learning rate

![LearningRate](README_images/LearningRate.png?raw=true)

Learning rate is the first hyperparameter that has been optimized. In general, constant values do not perform very well, as shown in the figure above (dotted lines). The loss of the validation data fluctuates significantly with the default value of 1e-3 and for larger learning rates (e.g. 1e-2). Smaller learning rate (e.g. 1e-4) works better, the loss function on train data decreases to below 0.1 and the loss function is around 0.4 for few epochs, then it increases a little bit (overfit).

Better results can be achieved gradually decreasing the learning rate during the training. In this case, an exponential decay has been used: *learning_rate=lr0 * decay_rate ^ (step/decay)*, with decay_rate=0.92. Similar results were obtained using lr0 = 1e-2 and 1e-3, decay = 35, with loss function on validation data stable around 0.24. A more aggressive decay (decay=100) leads to more overfitting, i.e. smaller loss on training data and bigger loss on validation data.

##### Padding

![Padding](README_images/Padding.png?raw=true)

An improvement of the loss on both training and validation data was achieved using same padding in the convolutional layers. Same padding helps to keep the information from the pixels close to the edges and avoids the shrinking of the images due to convolutional layers (assuming stride=1). At the same time, it increases significantly the number of trainable parameters from 1.5M to 2.7M.

##### Regularization

![regularization_](README_images/Regularization.png?raw=true)

flippa L2 and DP in legend
dotted line for no reg
standardize numbers

Regularization was used to reduce the overfitting. The figure above shows the impact of L2 and Dropout regularization. The regularization was applied only to the first fully connected layer after the stack of convolutional layers: experimental results showed that regularization on convolutional layers does not improve the loss. 
 
Without regularization, blue dotted line in the figure above, the loss after 10 epochs tends to zero, while the val-loss continues to oscillate. Dropout regularization stabilizes the val-loss and a rate of 15% shows the lowest val-loss. L2 regularization does not really improve the loss, it is negligible for very small regularization factors (e.g. 1e-4) or increase the loss for larger factors (e.g. 1e-3).  

#### Model training

The network has been trained with a GPU P5000, using Adam optimizer and binary crossentropy loss function. The learning rate has been decreased exponentially, from an initial value of 1e-3, with a decay step of 35 and decay rate of 0.92.

After 10 epochs (batches of 128 images), the train loss is stable around 0.105 and the validation loss is around 0.199, which correspond to a ROC AUC of 0.992 and 0.918 respectively.  

The training of the model is saved in notebook/ModelTraining.

#### Model Testing

The model has been tested on the dedicated test set, that showed a loss of 0.183, similar to the validation set. To convert the probability to class labels, an optimal threshold has been extracted from the validation set through the  expression: *optimal_threshold = argmin(TruePositiveRate - (1-FalsePositiveRate))* and used on both validation and test set. The optimal threshold results in the following metrics:

| | ROC_AUC | Precision | Recall | f1_score | f2_score |
|---|---|---|---|---|---|
| Validation | 0.918 | 0.390 | 0.842 | 0.533 | 0.684|
|Test | 0.921 | 0.366 | 0.824 | 0.507 | 0.659 |

#### Examples of correctly classified images

Here are a few examples of corrected classification on test data for both positive and negative examples. The title of each image indicates the actual class and the probability that the image contains a crack. Note that the optimal threshold, evaluated on validation data, is equal to 0.08 (i.e. p<0.078 --> Non-cracked, p>0.08 --> Cracked)

##### True Positive
![True Positive Examples](README_images/TruePositive.png?raw=true)

##### True Negative
![True Negative Examples](README_images/TrueNegative.png?raw=true)

#### Examples of misclassified images

Here are a few misclassified images for both positive and negative examples.

##### False Positive
![False Positive Examples](README_images/FalsePositive.png?raw=true)

##### False Negative
![False Negative Examples](README_images/FalseNegative.png?raw=true)

False positive examples show common features like stripes and granules. False negative examples show common features like very small and shallow cracks and potholes that look like granules/stains.

In general, many images were manually analyzed and in several cases it was very hard to classify them also for a person.  

## Repository description

notebooks/ contains an example of data augmentation (DataAugmentation), the training of the model (ModelTraining) and the testing of the trained model (ModelTesting)

app/ contains all the files to run the application: the trained model with the weights, the Flask application, the Dockerfile and the requirements file

test_images/ contains few images from the test subset that can be used to test the app

## Test the app

1) build the docker image: docker build -t crack_api .

2) create the image and start it: docker run -p 5000:5000 crack_api

3) go to http://0.0.0.0:5000/ and test the app

## References

[1] Structural Defects Network (SDNET) 2018, Concrete Cracks Image Dataset, https://www.kaggle.com/aniruddhsharma/structural-defects-network-concrete-crack-

[2] Very deep convolutional networks for large-scale image recognition, K. Simonyan and A. Zisserman, https://arxiv.org/pdf/1409.1556.pdf

[3] Batch Normalization: accelerating deep network training by reducing internal covariate shift, S. Ioffe and C. Szegedy, https://arxiv.org/pdf/1502.03167.pdf

[4] Understanding the Disharmony between Dropout and Batch Normalization by
Variance Shift, X. Li at al,  https://arxiv.org/pdf/1801.05134.pdf
