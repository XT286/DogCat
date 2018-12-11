# Dog-Cat Photo Classification with ResNet50

Our goal is to train a Neural Network model to classify photos of cats and dogs. We used transfer learning with ResNet50 to train the model. This was a Kaggle competition.

Linked to Data: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

I borrowed several constructing blocks for ResNet50 from https://www.kaggle.com/ryanmarfty/dogcat-res50. They are packed into https://github.com/XT286/DogCat/blob/master/resnet50.py.

We train the model with early stopping in https://github.com/XT286/DogCat/blob/master/dogcat_resnet_train_v2.py. 

We output the prediction using https://github.com/XT286/DogCat/blob/master/dogcat_resnet_test.py.

We then upload the prediction to Kaggle, and the current score is 0.07013. This score ranked 169/1314, approximately top 12% of all participants.

Update: in the third version of our model:
https://github.com/XT286/DogCat/blob/master/dogcat_resnet_train_v3.py
https://github.com/XT286/DogCat/blob/master/dogcat_resnet_test_v3.py
We combined two models InceptionV3 and Xception to make predictions. We reached a score of 0.05149, and ranked 74/1314, roughly top 5% of all participants.
