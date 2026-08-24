# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start the program and create a dataset containing input features and their corresponding binary class labels (0 or 1).
2.Initialize the weights, bias, learning rate, and number of iterations, then calculate the predicted probability using the Sigmoid function.
3.Calculate the error and gradients, and update the weights and bias using Gradient Descent repeatedly for the specified number of iterations.
4.Convert the predicted probabilities into class labels using a threshold of 0.5, then evaluate the model using accuracy and display the results.
5.Stop the program.
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: BRINDHA.M
RegisterNumber: 212225060038
*/
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create sample dataset
# X = Hours studied
# y = 0 = Fail, 1 = Pass

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Step 2: Standardize the input
X_mean = np.mean(X)
X_std = np.std(X)
X_scaled = (X - X_mean) / X_std

# Step 3: Initialize parameters
w = 0.0
b = 0.0
alpha = 0.1
epochs = 1000

n = len(X_scaled)
losses = []

# Step 4: Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 5: Gradient Descent
for i in range(epochs):

    # Calculate predicted probability
    z = w * X_scaled + b
    y_pred = sigmoid(z)

    # Calculate Log Loss
    loss = -np.mean(
        y * np.log(y_pred + 1e-9) +
        (1 - y) * np.log(1 - y_pred + 1e-9)
    )

    losses.append(loss)

    # Calculate gradients
    dw = (1 / n) * np.sum((y_pred - y) * X_scaled)
    db = (1 / n) * np.sum(y_pred - y)

    # Update weight and bias
    w = w - alpha * dw
    b = b - alpha * db

# Step 6: Make predictions
probabilities = sigmoid(w * X_scaled + b)

predictions = (probabilities >= 0.5).astype(int)

# Step 7: Calculate accuracy
accuracy = np.mean(predictions == y) * 100

print("Final Weight:", w)
print("Final Bias:", b)
print("Accuracy:", accuracy, "%")

print("\nActual Class:    ", y)
print("Predicted Class: ", predictions)

# Step 8: Predict for a new student
hours = 6.5

hours_scaled = (hours - X_mean) / X_std
probability = sigmoid(w * hours_scaled + b)

if probability >= 0.5:
    result = "Pass"
else:
    result = "Fail"

print("\nNew Student")
print("Hours Studied:", hours)
print("Probability:", probability)
print("Prediction:", result)

# Step 9: Plot Loss vs Iterations
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.title("Loss vs Iterations")

# Step 10: Plot Logistic Regression
plt.subplot(1, 2, 2)

plt.scatter(X, y, label="Actual Data")

X_line = np.linspace(min(X), max(X), 100)
X_line_scaled = (X_line - X_mean) / X_std

Y_line = sigmoid(w * X_line_scaled + b)

plt.plot(X_line, Y_line, label="Logistic Regression")

plt.axhline(0.5, linestyle="--", label="Decision Threshold")

plt.xlabel("Hours Studied")
plt.ylabel("Probability")
plt.title("Logistic Regression using Gradient Descent")
plt.legend()

plt.tight_layout()
plt.show()
```

## Output:
<img width="920" height="627" alt="image" src="https://github.com/user-attachments/assets/02360a45-7644-4412-afa6-a5c9fe65845e" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

