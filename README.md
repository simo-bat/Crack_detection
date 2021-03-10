# Surface cracks detection in pavements

Capstone project for the Springboard Machine Learning Engineering Career Track

## Problem definition



## Data

To train and test this model, the pavement data from SDNET2018 dataset (https://www.kaggle.com/aniruddhsharma/structural-defects-network-concrete-crack-images) has been used. The dataset contains images of pavements with (2600 images) and without cracks (21700 images). For mthe training, data augmentation has been used to balnce the two classes. First the data have been divided in train, validation and test, then, only for the train dataset, new images with cracks have been created and saved. An example of data augmentation is reported in notebooks/DataAugmentation.ipynb  

## Convolutional Neural Network architecture

The architecture used for this project is based on convolutional neural network (CNN) and it is inspired by the many architectures reported in literature, especially VGG16. It consists in the sequence of 5 CNN blocks, the first three blocks have a convolutional layer, followed by a batch normalization, ReLu activation function and a max pool layer. The last two blocks have two consequent convolutional+batch normalization+ReLu layers before the max pool. This approach increases the effective receptive field, limiting the number of trainable parameters and accelerating the trainng. 
The stack of convolutional layers is followed by two fully connected layers, the last activation function is a softmax, that perform the binary classification.
Regularization has been used to limit the overfitting: a dropout layer is inserted after the last batch normalization layer and 
To limit the overfittng, L2 and dropout regularization have been used in the first fully connected layer. Experimentation showed that no L2 regularization is needed for the convolutional layers.

### Training

For the training,  

## Repository description



### Docker container