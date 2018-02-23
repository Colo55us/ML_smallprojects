import keras
from keras.models import Sequential,Input,Model 
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU,ELU
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from sklearn import model_selection

batch_size = 64
n_epochs = 15
n_classes = 10

(train_X,train_y),(test_X,test_y) = fashion_mnist.load_data()

#print(train_y.size)

train_X = train_X.reshape(-1,28,28,1)
test_X = test_X.reshape(-1,28,28,1)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

train_X = train_X/255
test_X = test_X/255

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

train_X,valid_X ,train_label,Valid_label = model_selection.train_test_split(train_X,train_y,test_size=0.2)

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(ELU(alpha=0.1))
model.add(MaxPooling2D((2,2),padding='same'))
model.add(Conv2D(64,kernel_size=(3,3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(ELU(alpha=0.1))
model.add(MaxPooling2D((2,2),padding='same'))
model.add(Conv2D(128,kernel_size=(3,3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(ELU(alpha=0.1))
model.add(MaxPooling2D((2,2),padding='same'))
model.add(Flatten())
model.add(Dense(128,activation='linear'))
model.add(ELU(alpha=0.1))
model.add(Dense(n_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.summary()

model.fit(train_X,train_label,batch_size=batch_size,epochs=n_epochs,verbose=1,validation_data=(valid_X,Valid_label))

test_eval = model.evaluate(test_X,test_y,verbose=0)

print('Accuracy of the CNN model is ',test_eval[1])



