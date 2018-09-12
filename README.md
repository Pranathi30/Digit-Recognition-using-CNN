# MNIST Data Digit-Recognition-using-CNN
https://www.kaggle.com/pranathik/digit-recognizer


Validation Accuracy achieved is 0.99414 and currently standing in top 25% of the teams

1) used network:

a) 2 layers of Conv2D with 5X5 kernel size, 32 filters and same padding

b) Pooling layer of size 2X2 

c)drop out layer by randomly removing 25% of the weights

d) 2 layers of Conv2D with 5X5 kernel size, 64 filters and same padding

e) Pooling layer of size 2X2

f)drop out layer by randomly removing 25% of the weights

g) Flattening the output

h)adding a fully connected dense classfier which outputs 512 parameters

i)dropout layer which removed 50% of the weights (to reduce overfitting by randomly removing parameters)

j)adding a fully connected dense classfier with sigmoid activation and which outputs 10 softmax probabilities

2) I performed data augmentation to remove overfitting. Accuracy before data augmentation was ~0.98 and it significantly improved by performing data augmentation

3) validation loss and validation accuracy are both better than training loss and training accuracy in my final model


