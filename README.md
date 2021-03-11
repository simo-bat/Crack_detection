# Surface cracks detection in pavements

Capstone project for the Springboard Machine Learning Engineering Career Track

## Problem definition



## Data

SDNET2018 is an annotated dataset of concrete images with and without cracks from bridge decks, walls and pavements (https://www.kaggle.com/aniruddhsharma/structural-defects-network-concrete-crack-images). The pavements subset, that include 2600 positive images (with crack) and 21700 negative images (without crack),      has been used to train and test this model. First, the data have been divided into train (80%), validation (10%) and test (10%), then, only for the train subset, new images with cracks have been created and saved. An example of data augmentation is reported in notebooks/DataAugmentation.ipynb  

## Model

#### Convolutional Neural Network architecture

The architecture used for this project is based on convolutional neural network (CNN) and it is inspired by the many architectures reported in literature, especially VGG16. It consists in the sequence of 5 CNN blocks, the first three blocks have a convolutional layer, followed by a batch normalization, relu activation function and a max pool layer. The last two blocks have two consecutives convolutional+batch normalization+ReLu layers before the max pool. This approach allows to increase the effective receptive field, limiting the number of trainable parameters and accelerating the training.
The stack of convolutional layers is followed by two fully connected layers, with a final softmax activation function that performs the binary classification.

L2 and dropout regularization have been used in the first fully connected layer to limit the overfitting. Dropout has been used only after the last batch normalization layer to avoid variance shift (https://arxiv.org/pdf/1801.05134.pdf). Experimentation showed that adding L2 regularization to the convolutional layers does not improve the performances.

#### Training

The network has been trained with a GPU P5000 for 10 epochs, using Adam optimizer and batches of 128 images. The learning rate has been decreased exponentially, from an initial value of 1e-3, with a decay step of 35 and decay rate of 0.92.
 
The training of the model is saved in notebook/Model.

## Repository description

notebooks/ contains an example of data augmentation (DataAugmentation), the training of the model (Model), some hyperparameters tuning (HyperparametersDependences) and the testing of the trained model (FinalModelTesting)

App/ contains all the files to run the application: the trained model with the weights, the Flask application, the Dockerfile and the requirements

### Docker container

To test the app, first build the docker image:

docker build -t crack_api .

then create the image and start it:

docker run -p 5000:5000 crack_api

finally go to http://0.0.0.0:5000/
