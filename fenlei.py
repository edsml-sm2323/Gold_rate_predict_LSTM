import keras
from keras . datasets import mnist
from keras . models import Sequential
from keras . layers import Dense, Dropout
from keras . optimizers import SGD
import pandas as pd
import numpy as np
from sklearn . preprocessing import StandardScaler, OneHotEncoder
from sklearn . model_selection import train_test_split
from sklearn.compose import ColumnTransformer
batch_size = 128
num_classes = 2
epochs = 20
df = pd.read_csv("jyc48-16.csv")
df = df.dropna()
df.iloc[:, 0] = range(len(df))
y = df.loc[:, 'RAIN']
X = df.drop('RAIN', axis=1)
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=0.05)
x_scaler = StandardScaler()
x_train = x_scaler . fit_transform(x_train)
x_test = x_scaler . transform(x_test)
print(x_train.shape)
y_train = y_train .values. reshape((-1, 1))
y_test = y_test .values. reshape((-1, 1))



"""
print(y_train.shape)

preprocessor = ColumnTransformer(
    transformers=[
                ('cat', OneHotEncoder(), ['RAIN'])
    ])

y_train = preprocessor . fit_transform(y_train)

y_test = preprocessor . transform(y_test)
"""

y_train = keras . utils . to_categorical(y_train, num_classes)
y_test = keras . utils . to_categorical(y_test, num_classes)

print(y_train)
model = Sequential()

# And then sequentially add new layers .
# A Dense layer is the one we covered this chapter ,
# where a neuron connects to all the neurons in ,
# following layer .
# For each layer , we have to specify the activation ,
# function and the output size . In the first layer ,
# we also have to specify the input shape .

model . add ( Dense (512 , activation = 'relu' ,
                     input_shape =(4 ,)))

# Dropout is a regularization technique ( to prevent
# overfitting )
model . add ( Dropout (0.2))
model . add ( Dense (512 , activation = 'relu' ))
model . add ( Dropout (0.2))
model . add ( Dense ( num_classes , activation = 'softmax' ))

model . compile ( loss = 'categorical_crossentropy' ,
                  optimizer = SGD () ,
                  metrics =[ 'accuracy' ])

# After the network is compiled we can train it , using
# our training set .
history = model . fit ( x_train , y_train ,
                        batch_size=batch_size ,
                        epochs=epochs,
                        verbose=1,
                        validation_split =0.1)

# Finally , we check the performance of the model
# in the test set
score = model . evaluate ( x_test , y_test , verbose =0)
print ( 'Test loss :' , score [0])
print ( 'Test accuracy :' , score [1])

