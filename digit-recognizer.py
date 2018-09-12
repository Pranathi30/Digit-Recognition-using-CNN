# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


from keras import layers
from keras.layers import Dropout
from keras import models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras import regularizers

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train.head()
test.head()

train_y= train['label']
train_X= train.drop(labels=['label'],axis=1)

del train

sns.countplot(train_y)
train_X.columns[train_X.isnull().any()]
test.columns[test.isnull().any()]

#normalizing 
train_X= train_X/255
test=test/255

train_X= train_X.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

#one hot encode labels
train_y=to_categorical(train_y)

#split into train and validation sets
random_seed=2
train_X,val_X,train_y,val_y=train_test_split(train_X,train_y,test_size=0.2,random_state=random_seed)


train_X.shape
val_X.shape

train_y.shape
val_y.shape


train_X[0]

plt.imshow(train_X[0][:,:,0])

model3 = models.Sequential()
model3.add(layers.Conv2D(32,(5,5),padding='same',activation='relu',input_shape=(28,28,1)))
model3.add(layers.Conv2D(32,(5,5),padding='same',activation='relu'))
model3.add(layers.MaxPooling2D(2,2))
model3.add(Dropout(0.25))
model3.add(layers.Conv2D(64,(5,5),padding='same',activation='relu'))
model3.add(layers.Conv2D(64,(5,5),padding='same',activation='relu'))
model3.add(layers.MaxPooling2D(2,2))
model3.add(Dropout(0.25))
model3.add(layers.Flatten())
model3.add(layers.Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model3.add(Dropout(0.5))
model3.add(layers.Dense(10,activation='softmax',kernel_regularizer=regularizers.l2(0.001)))

datagen = ImageDataGenerator(rotation_range=10,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=False,vertical_flip=False,fill_mode='nearest')
model3.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
history3 = model3.fit_generator(datagen.flow(train_X,train_y,batch_size=100),verbose=2,steps_per_epoch=336,epochs=30,validation_data=(val_X,val_y),validation_steps=50)

history3.history.keys()
acc3 = history3.history['acc']
val_acc3 = history3.history['val_acc']
loss3 = history3.history['loss']
val_loss3 = history3.history['val_loss']
epochs = range(1, len(acc3) + 1)
plt.plot(epochs, acc3, 'bo', label='Training acc')
plt.plot(epochs, val_acc3, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss3, 'bo', label='Training loss')
plt.plot(epochs, val_loss3, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

loss_val,acc_val = model3.evaluate(val_X,val_y)
acc_val

test_predictions=model3.predict(test)

result = np.argmax(test_predictions,axis=1)
result=pd.Series(result,name='Label')
#result
sub = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)
sub.head()
sub.to_csv("sub_5.csv",index=False)























