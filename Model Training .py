#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[ ]:


dataset=r'C:\facemask detection\dataset'
imagePaths=list(paths.list_images(dataset))


# In[ ]:


imagePaths


# In[ ]:


data=[]
labels=[]

for i in imagePaths:
    label=i.split(os.path.sep)[-2]
    labels.append(label)
    image=load_img(i,target_size=(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    data.append(image)


# In[ ]:


data


# In[ ]:


labels


# In[ ]:



data=np.array(data,dtype='float32')
labels=np.array(labels)


# In[ ]:


data.shape


# In[ ]:


labels


# In[ ]:



lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)


# In[ ]:


labels


# In[ ]:


train_X,test_X,train_Y,test_Y=train_test_split(data,labels,test_size=0.20,stratify=labels,random_state=10)


# In[ ]:


train_X.shape


# In[ ]:


train_Y.shape


# In[ ]:


test_X.shape


# In[ ]:


test_Y.shape


# In[ ]:



aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,vertical_flip=True,fill_mode='nearest')


# In[ ]:



baseModel=MobileNetV2(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))


# In[ ]:


baseModel.summary()


# In[ ]:


headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name='Flatten')(headModel)
headModel=Dense(128,activation='relu')(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation='softmax')(headModel)

model=Model(inputs=baseModel.input,outputs=headModel)


# In[ ]:



for layer in baseModel.layers:
    layer.trainable=False


# In[ ]:





# In[ ]:


learning_rate=0.001

EPOCHS=10
BS=10

opt=Adam(lr=learning_rate,decay=learning_rate/EPOCHS)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

H=model.fit(
    aug.flow(train_X,train_Y,batch_size=BS),
    steps_per_epoch=len(train_X)//BS,
    validation_data=(test_X,test_Y),
    validation_steps=len(test_X)//BS,
    epochs=EPOCHS
)


# In[ ]:


model.save('my_model.h5')


# In[ ]:


predict=model.predict(test_X,batch_size=BS)
predict=np.argmax(predict,axis=1)
print(classification_report(test_Y.argmax(axis=1),predict,target_names=lb.classes_))


# In[ ]:



from matplotlib import pyplot as plt

plt.plot(H.history['loss'],'r',label='training loss')
plt.plot(H.history['val_loss'],label='validation loss')
plt.title('MODEL LOSS')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[ ]:



plt.plot(H.history['accuracy'],'r',label='training accuracy')
plt.plot(H.history['val_accuracy'],label='validation accuracy')
plt.title('MODEL ACCURACY')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




