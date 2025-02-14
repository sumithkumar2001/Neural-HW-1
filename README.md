# Neural-HW-1
Sumith kumar kotagiri (700759479)


1.	Tensor Manipulations & Reshaping
Explanation :

## Overview
This Python script demonstrates various tensor operations using NumPy. The script performs the following steps:
1. Creates a random tensor of shape (4,6).
2. Determines and prints the rank (number of dimensions) and shape of the tensor.
3. Reshapes the tensor to (2,3,4) and then transposes it to (3,2,4).
4. Creates a smaller tensor of shape (1,4) and broadcasts it to match the transposed tensor.
5. Adds the broadcasted tensor to the transposed tensor and prints the resulting shape.
## Prerequisites
Ensure you have Python installed along with the NumPy library. You can install NumPy using:
```sh
pip install numpy
```
## Code Explanation
### Step 1: Creating a Random Tensor
```python
import numpy as np
tensor = np.random.uniform(size=(4, 6))
```
- A tensor of shape (4,6) is created with random values between 0 and 1.
### Step 2: Finding Rank and Shape
```python
rank = np.ndim(tensor)
shape = np.shape(tensor)
print(f"Original Tensor Rank: {rank}, Shape: {shape}")
```
- The rank (number of dimensions) and shape of the tensor are computed and printed.
### Step 3: Reshaping and Transposing the Tensor
```python
reshaped_tensor = np.reshape(tensor, (2, 3, 4))
transposed_tensor = np.transpose(reshaped_tensor, (1, 0, 2))
```
- The tensor is reshaped from (4,6) to (2,3,4).
- The reshaped tensor is transposed to (3,2,4), changing the order of axes.
### Step 4: Broadcasting and Addition
```python
small_tensor = np.random.uniform(size=(1, 4))
broadcasted_tensor = np.broadcast_to(small_tensor, (3, 2, 4))
added_tensor = transposed_tensor + broadcasted_tensor
```
- A smaller tensor of shape (1,4) is created.
- It is broadcasted to shape (3,2,4) to match the transposed tensor.
- Element-wise addition is performed.
### Output
The script prints the shapes at various stages, confirming that operations were performed successfully.
## Running the Script
Simply execute the script in a Python environment:
```sh
python script.py
```
## Expected Output Format
```
Original Tensor Rank: 2, Shape: (4, 6)
Reshaped Tensor Shape: (2, 3, 4)
Transposed Tensor Shape: (3, 2, 4)
Broadcasted Tensor Shape: (3, 2, 4)
Resulting Tensor Shape after Addition: (3, 2, 4)
```



2.	Loss Functions & Hyperparameter Tuning
Explanation :

## Overview
This Python script demonstrates the computation and visualization of two common loss functions in machine learning:
- **Mean Squared Error (MSE)**: Measures the average squared difference between true values and predicted values.
- **Categorical Cross-Entropy (CCE)**: Commonly used in classification problems, measuring how well the predicted probabilities match the true labels.
The script evaluates these loss functions for two sets of model predictions and compares the results using a bar chart.
## Prerequisites
Ensure you have Python installed along with the required libraries. You can install them using:
```sh
pip install numpy matplotlib scikit-learn
```
## Code Explanation
### Step 1: Define True Values and Predictions
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, log_loss
y_true = np.array([0, 1, 1, 0, 1])
y_pred1 = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
y_pred2 = np.array([0.2, 0.8, 0.7, 0.3, 0.6])
```
- `y_true` represents the actual class labels.
- `y_pred1` and `y_pred2` are two sets of predicted probabilities.
### Step 2: Compute MSE and CCE Losses
```python
mse1 = mean_squared_error(y_true, y_pred1)
cce1 = log_loss(y_true, y_pred1)
mse2 = mean_squared_error(y_true, y_pred2)
cce2 = log_loss(y_true, y_pred2)
```
- The MSE and CCE are calculated for both sets of predictions.
- These values quantify the accuracy of the predictions.
### Step 3: Print Loss Values
```python
print(f"MSE for y_pred1: {mse1}, CCE for y_pred1: {cce1}")
print(f"MSE for y_pred2: {mse2}, CCE for y_pred2: {cce2}")
```
- Prints the computed loss values for comparison.
### Step 4: Visualizing the Loss Values
```python
labels = ['MSE y_pred1', 'CCE y_pred1', 'MSE y_pred2', 'CCE y_pred2']
values = [mse1, cce1, mse2, cce2]
plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'orange', 'blue', 'orange'])
plt.ylabel('Loss Value')
plt.title('Comparison of MSE and CCE Loss')
plt.show()
```
- A bar chart is generated to visually compare the loss values.
## Running the Script
Execute the script in a Python environment:
```sh
python script.py
```
## Expected Output Format
```
MSE for y_pred1: <value>, CCE for y_pred1: <value>
MSE for y_pred2: <value>, CCE for y_pred2: <value>
```
Additionally, a bar chart will be displayed, comparing the loss values.



3.	Train a Model with Different Optimizers
Explanation :

## Overview
This Python script compares the performance of two different optimizers, **Adam** and **SGD**, on the MNIST-like **Digits dataset** using a simple neural network classifier. The script evaluates and visualizes the training and validation accuracy of both optimizers.
## Prerequisites
Ensure you have Python installed along with the required libraries. You can install them using:
```sh
pip install numpy matplotlib scikit-learn
```
## Code Explanation
### Step 1: Load and Preprocess the Dataset
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
digits = load_digits()
X, y = digits.data / 16.0, digits.target  # Normalize data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Loads the **Digits dataset** (an alternative to MNIST).
- Normalizes the feature values by dividing by 16.
- Splits the dataset into **training (80%)** and **testing (20%)** sets.
- Uses **StandardScaler** for feature standardization.
### Step 2: Train Models with Adam and SGD Optimizers
```python
from sklearn.neural_network import MLPClassifier
adam_model = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=20, random_state=42)
sgd_model = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=20, random_state=42)
adam_model.fit(X_train, y_train)
sgd_model.fit(X_train, y_train)
```
- Defines two neural networks with one hidden layer of **128 neurons**.
- Uses **ReLU** activation.
- Trains one model using **Adam** optimizer and the other using **SGD**.
- Limits training to **20 iterations** for both models.
### Step 3: Compute Training and Validation Accuracy
```python
from sklearn.metrics import accuracy_score
adam_train_acc = accuracy_score(y_train, adam_model.predict(X_train))
adam_val_acc = accuracy_score(y_test, adam_model.predict(X_test))
sgd_train_acc = accuracy_score(y_train, sgd_model.predict(X_train))
sgd_val_acc = accuracy_score(y_test, sgd_model.predict(X_test))
```
- Evaluates both models on the **training** and **validation** datasets.
- Computes **accuracy scores** for both optimizers.
### Step 4: Visualize Accuracy Comparison
```python
import matplotlib.pyplot as plt
labels = ['Adam Train', 'Adam Val', 'SGD Train', 'SGD Val']
values = [adam_train_acc, adam_val_acc, sgd_train_acc, sgd_val_acc]
plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'orange', 'blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Comparison of Adam vs. SGD on Digits Dataset')
plt.show()
```
- Plots a **bar chart** comparing training and validation accuracy for both optimizers.
## Running the Script
To execute the script, simply run:
```sh
python script.py
```
## Expected Output
The script prints accuracy values for training and validation datasets and generates a bar chart. Example format:
```
Adam Train Accuracy: 0.99, Adam Validation Accuracy: 0.94
SGD Train Accuracy: 0.95, SGD Validation Accuracy: 0.90
```
Additionally, a **bar chart** is displayed comparing accuracy values.



4.	Train a Neural Network and Log to TensorBoard
Explanation :

## Overview
This Python script compares the performance of two different optimizers, **Adam** and **SGD**, on the MNIST-like **Digits dataset** using a simple neural network classifier. The script evaluates and visualizes the training and validation accuracy of both optimizers.
## Prerequisites
Ensure you have Python installed along with the required libraries. You can install them using:
```sh
pip install numpy matplotlib scikit-learn
```
## Code Explanation
### Step 1: Load and Preprocess the Dataset
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
digits = load_digits()
X, y = digits.data / 16.0, digits.target  # Normalize data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Loads the **Digits dataset** (an alternative to MNIST).
- Normalizes the feature values by dividing by 16.
- Splits the dataset into **training (80%)** and **testing (20%)** sets.
- Uses **StandardScaler** for feature standardization.
### Step 2: Train Models with Adam and SGD Optimizers
```python
from sklearn.neural_network import MLPClassifier
adam_model = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=20, random_state=42)
sgd_model = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=20, random_state=42)
adam_model.fit(X_train, y_train)
sgd_model.fit(X_train, y_train)
```
- Defines two neural networks with one hidden layer of **128 neurons**.
- Uses **ReLU** activation.
- Trains one model using **Adam** optimizer and the other using **SGD**.
- Limits training to **20 iterations** for both models.
### Step 3: Compute Training and Validation Accuracy
```python
from sklearn.metrics import accuracy_score
adam_train_acc = accuracy_score(y_train, adam_model.predict(X_train))
adam_val_acc = accuracy_score(y_test, adam_model.predict(X_test))
sgd_train_acc = accuracy_score(y_train, sgd_model.predict(X_train))
sgd_val_acc = accuracy_score(y_test, sgd_model.predict(X_test))
```
- Evaluates both models on the **training** and **validation** datasets.
- Computes **accuracy scores** for both optimizers.
### Step 4: Visualize Accuracy Comparison
```python
import matplotlib.pyplot as plt
labels = ['Adam Train', 'Adam Val', 'SGD Train', 'SGD Val']
values = [adam_train_acc, adam_val_acc, sgd_train_acc, sgd_val_acc]
plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'orange', 'blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Comparison of Adam vs. SGD on Digits Dataset')
plt.show()
```
- Plots a **bar chart** comparing training and validation accuracy for both optimizers.
## Running the Script
To execute the script, simply run:
```sh
python script.py
```
## Expected Output
The script prints accuracy values for training and validation datasets and generates a bar chart. Example format:
```
Adam Train Accuracy: 0.99, Adam Validation Accuracy: 0.94
SGD Train Accuracy: 0.95, SGD Validation Accuracy: 0.90
```
Additionally, a **bar chart** is displayed comparing accuracy values.
