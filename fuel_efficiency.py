import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import smp
#a=smp.fun(2,5)
#print(a)

dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                 'Acceleration',  'Model Year','Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

print("hello")

dataset = raw_dataset.copy()
train_stats = dataset.describe()
# train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)
dataset.tail()

print(dataset)

df=dataset
print(df)

# create an instance of SimpleImputer to replace missing values with the mean
imputer = SimpleImputer(strategy='mean')

# fit the imputer to the data
imputer.fit(df)

# transform the data with the imputer to replace missing values
df_imputed = imputer.transform(df)

# convert the numpy array back to a dataframe
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
# dataset=df_imputed
data=df_imputed
print(data)

duplicates = data[data.duplicated(data.columns[:-1], keep='last')]

# Remove the duplicate rows from the original dataframe
data.drop_duplicates(subset=data.columns[:-1], keep='first', inplace=True)
print(data)

# Separate the features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # split remaining 60% into training and validation

def build_model():
  # Create a deep learning model
  model = Sequential()
  model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
  # model.add(Dropout(0.2))
  model.add(Dense(8,activation='relu'))
  # model.add(Dropout(0.2))
  model.add(Dense(1))
  # model.add(Dropout(0.2))
  
  dllr = Adam(learning_rate=0.001)
  model.compile(loss='mse', optimizer=dllr, metrics=['mae'])
  return model
model=build_model()

# Train the model
history = model.fit(X_train, y_train, epochs=3000, batch_size=32, validation_data=(X_val, y_val))

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  plt.show()
plot_history(history)

# arr=np.array(l)
test_data = pd.DataFrame({
    'Cylinders': [8,8],
    'Displacement': [307,350],
    'Horsepower':[130,165],
    'Weight': [3504,3693],
    'Acceleration': [12,11.5],
    'Model Year': [70,70],
    'Origin':[1.0,1.0]
})
predictions = model.predict(test_data)
out=smp.displayoutput(test_data,predictions)
print(out)

model = build_model()
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print(epoch, end=' ')

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=1)

history = model.fit(X_train, y_train, epochs=3000,
                    validation_split = 0.2, verbose=2,validation_data=(X_val, y_val), callbacks=[early_stop, PrintDot()])

plot_history(history)

arr=np.array([[8,307.0,130.0,3504.0,12.0,70,1.0]])
test_data = pd.DataFrame({
    'Cylinders': [8,8],
    'Displacement': [307,350],
    'Horsepower':[130,165],
    'Weight': [3504,3693],
    'Acceleration': [12,11.5],
    'Model Year': [70,70],
    # 'USA':[1.0],
    # 'Europe':[0.0],
    # 'Japan':[0.0]
    'Origin':[1.0,1.0]
})
predictions = model.predict(test_data)
out=predictions
print(out)

