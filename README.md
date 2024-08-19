# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Joyce Beulah R
### Register Number: 212222230058

```python
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
```
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

ws = gc.open('demo').sheet1

rows = ws.get_all_values()
```
```python
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'sno':'float'})
df = df.astype({'marks':'float'})
df.head()

x = df[["sno"]].values
y = df[["marks"]].values
```
```python
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 33)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train1 = scaler.transform(x_train)
```
```python
marks_data = Sequential([Dense(6,activation='relu'),Dense(7,activation='relu'),Dense(1)])
marks_data.compile(optimizer = 'rmsprop' , loss = 'mse')

marks_data.fit(x_train1 , y_train,epochs = 500)

loss_df = pd.DataFrame(marks_data.history.history)
loss_df.plot()

x_test1 = scaler.transform(x_test)
marks_data.evaluate(x_test1,y_test)

X_n1 = [[30]]
X_n1_1 = scaler.transform(X_n1)
marks_data.predict(X_n1_1)
```
## Dataset Information

Include screenshot of the dataset

## OUTPUT

### Training Loss Vs Iteration Plot

Include your plot here

### Test Data Root Mean Squared Error

Find the test data root mean squared error

### New Sample Data Prediction

Include your sample input and output here

## RESULT

Include your result here
