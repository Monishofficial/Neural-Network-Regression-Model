# EXP-1 Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model
![image](https://github.com/user-attachments/assets/84093ee0-48a5-4bd2-b78d-5d8ee258d189)


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
### Name: MONISH N
### Register Number:212223240097
```python
import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.history={'loss':[]}

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


monish_brain=NeuralNetwork()
criteria=nn.MSELoss()
optimizer=optim.RMSprop(monish_brain.parameters(), lr=0.001)



def train_model(monish_brain,X_train,y_train,criteria,optimizer,epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      output = monish_brain(X_train)
      loss = criteria(output, y_train)
      loss.backward()
      optimizer.step()

      monish_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
## Dataset Information

<img width="243" height="672" alt="image" src="https://github.com/user-attachments/assets/f835f242-8529-424b-b5b1-4eb5b64dc04b" />


## OUTPUT

### Training Loss Vs Iteration Plot
<img width="659" height="471" alt="image" src="https://github.com/user-attachments/assets/026cd633-2b1a-4f99-aea6-a254f0136aee" />


### New Sample Data Prediction
<img width="778" height="115" alt="image" src="https://github.com/user-attachments/assets/124f27b8-255c-4a34-a0af-438d00b01982" />

## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.







