import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import smp
import re
import tkinter as tk
from tkinter import messagebox


# Create the welcome window
welcome_window = tk.Tk()

# Set the welcome window title
welcome_window.title("Welcome to Fuel Efficiency Predictor")

# Set the welcome window size
welcome_window.geometry("400x200")

# Add a label with the welcome message
welcome_label = tk.Label(welcome_window, text="Welcome to Fuel Efficiency Predictor!")
welcome_label.pack(pady=20)

# Add a button to start the program
def start_program():
    # Close the welcome window
    welcome_window.destroy()
    
start_button = tk.Button(welcome_window, text="Start", command=start_program)
start_button.pack()

# Run the welcome window's event loop
welcome_window.mainloop()

dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                 'Acceleration',  'Model Year','Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

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
  model.add(Dense(1,activation='sigmoid'))
  # model.add(Dropout(0.2))
  
  dllr = Adam(learning_rate=0.001)
  model.compile(loss='binary_crossentropy', optimizer=dllr, metrics=['mae'])
  return model

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
  plt.show(block=False)

model=build_model()
    
# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
plot_history(history)

def newmodel():
  new_model = build_model()
  class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
      if epoch % 100 == 0: print('')
      print(epoch, end=' ')

# The patience parameter is the amount of epochs to check for improvement
  early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=1)

  history = new_model.fit(X_train, y_train, epochs=100,
                    validation_split = 0.2, verbose=2,validation_data=(X_val, y_val), callbacks=[early_stop, PrintDot()])
  plot_history(history)
  return new_model


# Define a function to create a new error window
def create_error_window():
    # Create a new top-level window
    error_window = tk.Toplevel()

    # Set the dimensions and title of the new window
    error_window.geometry("200x100")
    error_window.title("Error")

    # Add a label to the new window
    label = tk.Label(error_window, text="Invalid input.")
    label.pack()
    

# Create a Tkinter GUI for the Auto MPG predictor

def datainput():
    # Get the input data from the entry fields
    cylinders = cylinders_entry.get().strip()
    displacement = displacement_entry.get().strip()
    horsepower = horsepower_entry.get().strip()
    weight = weight_entry.get().strip()
    acceleration = acceleration_entry.get().strip()
    model_year = model_year_entry.get().strip()
    origin = origin_entry.get().strip()

    # Check which fields are missing
    '''missing_fields = []
    if cylinders == "Enter an integer":
        missing_fields.append("cylinders")
    if displacement == "Enter an integer":
        missing_fields.append("displacement")
    if horsepower == "Enter an integer":
        missing_fields.append("horsepower")
    if weight == "Enter an integer":
        missing_fields.append("weight")
    if acceleration == "Enter an integer":
        missing_fields.append("acceleration")
    if model_year == "Enter an integer":
        missing_fields.append("model Year")
    if origin == "Enter an integer":
        missing_fields.append("origin")

    # Display a pop-up message if any fields are missing
    if missing_fields:
        message = "Please fill in the following fields:\n\n" + "\n".join(missing_fields)
        messagebox.showerror("Error", message)
        return
    for field_name in missing_fields:
        field_entry = globals()[field_name.lower() + "_entry"]
        field_entry.delete(0, END)
        field_entry.insert(0, "Enter an integer")
        field_entry.focus_set()
        return'''

    # Define regular expressions for integers and floating point numbers
    int_regex = r'^[+-]?\d+$'
    float_regex = r'^[+-]?\d+(\.\d*)?$'

    # Validate the input data
    if not re.match(int_regex, cylinders) or not re.match(int_regex, displacement) or not re.match(int_regex, horsepower) or not re.match(int_regex, weight) or not re.match(float_regex, acceleration) or not re.match(int_regex, model_year) or not re.match(int_regex, origin):
        create_error_window()
        return

    # Convert the input data to floats
    cylinders = float(cylinders)
    displacement = float(displacement)
    horsepower = float(horsepower)
    weight = float(weight)
    acceleration = float(acceleration)
    model_year = float(model_year)
    origin = float(origin)
    

    # Create a numpy array from the input data
    input_data = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]])
    return input_data
  

# Define a function to handle the button click event
def predict():
    input_data=datainput()
    
    
    # Use the trained model to make a prediction
    prediction = model.predict(input_data)
    prediction = smp.displayoutput(input_data,prediction)
    # Update the result label with the prediction
    result_label.config(text=f"The predicted MPG is: {prediction:.2f}")

def repredict():
  # Use the trained model to make a prediction
  new_model=newmodel()
  
  # Get the input data from the entry fields
  cylinders = float(cylinders_entry.get())
  displacement = float(displacement_entry.get())
  horsepower = float(horsepower_entry.get())
  weight = float(weight_entry.get())
  acceleration = float(acceleration_entry.get())
  model_year = float(model_year_entry.get())
  origin = float(origin_entry.get())

    # Create a numpy array from the input data
  input_data = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]])
  prediction = new_model.predict(input_data)
  prediction = smp.displayoutput(input_data,prediction)
  # Update the result label with the prediction
    
  result2_label.config(text=f"The predicted MPG is: {prediction:.2f}")

  
# Create the Tkinter window
window = tk.Tk()
window.geometry("1000x700")
window.title("Auto MPG Predictor")

heading_label = tk.Label(window, text="Enter the data", font=("Helvetica", 16, "bold"))
heading_label.grid(row=0, column=25, columnspan=2, pady=10)

# Add the entry fields

# Add a label and entry field for the cylinders input
cylinders_label = tk.Label(window, text="Cylinders:")
cylinders_label.grid(row=1, column=0)
cylinders_entry = tk.Entry(window)
cylinders_entry.grid(row=1, column=1)

# Insert a default message in the cylinders entry field
cylinders_entry.insert(0, "Enter an integer")

# Clear the cylinders entry field when the user clicks on it
def clear_cylinders_entry(event):
    cylinders_entry.delete(0, 'end')

cylinders_entry.bind('<Button-1>', clear_cylinders_entry)

displacement_label = tk.Label(window, text="Displacement:")
displacement_label.grid(row=2, column=0)
displacement_entry = tk.Entry(window)
displacement_entry.grid(row=2, column=1)
# Insert a default message in the displacement entry field
displacement_entry.insert(0, "Enter an integer")

# Clear the displacement entry field when the user clicks on it
def clear_displacement_entry(event):
    displacement_entry.delete(0, 'end')

displacement_entry.bind('<Button-1>', clear_displacement_entry)

horsepower_label = tk.Label(window, text="Horsepower:")
horsepower_label.grid(row=3, column=0)
horsepower_entry = tk.Entry(window)
horsepower_entry.grid(row=3, column=1)
# Insert a default message in the horsepower entry field
horsepower_entry.insert(0, "Enter an integer")

# Clear the horsepower entry field when the user clicks on it
def clear_horsepower_entry(event):
    horsepower_entry.delete(0, 'end')

horsepower_entry.bind('<Button-1>', clear_horsepower_entry)


weight_label = tk.Label(window, text="Weight:")
weight_label.grid(row=4, column=0)
weight_entry = tk.Entry(window)
weight_entry.grid(row=4, column=1)
weight_entry.insert(0, "Enter an integer")

# Clear the weight entry field when the user clicks on it
def clear_weight_entry(event):
    weight_entry.delete(0, 'end')

weight_entry.bind('<Button-1>', clear_weight_entry)

acceleration_label = tk.Label(window, text="Acceleration:")
acceleration_label.grid(row=5, column=0)
acceleration_entry = tk.Entry(window)
acceleration_entry.grid(row=5, column=1)
# Insert a default message in the acceleration entry field
acceleration_entry.insert(0, "Enter an integer")

# Clear the acceleration entry field when the user clicks on it
def clear_acceleration_entry(event):
    acceleration_entry.delete(0, 'end')

acceleration_entry.bind('<Button-1>', clear_acceleration_entry)


model_year_label = tk.Label(window, text="Model Year:")
model_year_label.grid(row=6, column=0)
model_year_entry = tk.Entry(window)
model_year_entry.grid(row=6, column=1)
# Insert a default message in the model year entry field
model_year_entry.insert(0, "Enter an integer")

# Clear the model year entry field when the user clicks on it
def clear_model_year_entry(event):
    model_year_entry.delete(0, 'end')

model_year_entry.bind('<Button-1>', clear_model_year_entry)

origin_label = tk.Label(window, text="Origin:")
origin_label.grid(row=7, column=0)
origin_entry = tk.Entry(window)
origin_entry.grid(row=7, column=1)
# Insert a default message in the origin entry field
origin_entry.insert(0, "Enter an integer")

# Clear the origin entry field when the user clicks on it
def clear_origin_entry(event):
    origin_entry.delete(0, 'end')

origin_entry.bind('<Button-1>', clear_origin_entry)

# Add the button
predict_button = tk.Button(window, text="Predict", command=predict)
predict_button.grid(row=8, column=0, columnspan=2)

predict2_button = tk.Button(window , text="Predict again",command=repredict)
predict2_button.grid(row=11, column=0, columnspan=2)


# Add the result label
result_label = tk.Label(window, text="")
result_label.grid(row=9, column=1, columnspan=2)

result2_label=tk.Label(window, text="")
result2_label.grid(row=12,column=1,columnspan=2)

# Run the Tkinter event loop
window.mainloop()
