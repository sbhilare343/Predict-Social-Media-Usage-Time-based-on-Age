# Predict-Social-Media-Usage-Time-based-on-Age

### 1. **Importing Libraries**
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
```
- **NumPy** (`np`): Used for numerical computations (e.g., calculating root mean squared error).
- **Pandas** (`pd`): Used for data manipulation and handling (e.g., reading data from CSV, describing it).
- **scikit-learn** (`sklearn`): Provides tools for splitting data, creating the Linear Regression model, and evaluating it using metrics.
- **Matplotlib** & **Seaborn**: Libraries for data visualization.

### 2. **Loading and Summarizing Data**
```python
df = pd.read_csv('/content/social-media.csv')
df.describe()

df.head()

df.isnull().sum()
```
- `pd.read_csv()`: Loads the dataset from a CSV file into a Pandas DataFrame.
- `df.describe()`: Provides summary statistics for the numerical columns (e.g., count, mean, standard deviation).
- `df.head()`: Displays the first few rows of the DataFrame.
- `df.isnull().sum()`: Checks for missing values in each column of the DataFrame.

### 3. **Feature and Target Selection**
```python
X = df.drop(['UsageDuraiton','TotalLikes', 'UserId','Country'], axis=1)
y = df['UsageDuraiton']
```
- **X**: Defines the input features for the model. It drops the columns that are irrelevant for the prediction (`UsageDuraiton`, `TotalLikes`, `UserId`, and `Country`) because they are either the target variable or non-numerical.
- **y**: Defines the target variable, which is **UsageDuraiton**. This is the value you want to predict.

### 4. **Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **train_test_split**: Splits the dataset into training and testing sets.
  - **X_train, y_train**: Training data (80% of the original dataset).
  - **X_test, y_test**: Testing data (20% of the original dataset).
  - `test_size=0.2`: Specifies that 20% of the data will be used for testing.
  - `random_state=42`: Ensures reproducibility by setting a seed for the random splitting.

### 5. **Model Training**
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
- **LinearRegression()**: Initializes a linear regression model.
- `model.fit(X_train, y_train)`: Trains the model using the training data (`X_train` and `y_train`).

### 6. **Model Evaluation**
```python
MAE = mean_absolute_error(y_test, model.predict(X_test))
MSE = mean_squared_error(y_test, model.predict(X_test))
RMSE = np.sqrt(MSE)
R2 = r2_score(y_test, model.predict(X_test))

print('MAE:', MAE)
print('MSE:', MSE)
print('RMSE:', RMSE)
print('R2:', R2)
```
- **mean_absolute_error**: Calculates the average absolute difference between predicted and actual values (`MAE`).
- **mean_squared_error**: Measures the average squared difference between predicted and actual values (`MSE`).
- **np.sqrt(MSE)**: Computes the root mean squared error (`RMSE`), which provides an error metric in the same units as the target variable.
- **r2_score**: Computes the coefficient of determination (`R²`), which explains how well the regression predictions approximate the real data.
- The evaluation metrics are printed for analysis:
  - **MAE**: Mean Absolute Error.
  - **MSE**: Mean Squared Error.
  - **RMSE**: Root Mean Squared Error.
  - **R²**: Indicates the proportion of variance explained by the model (1 = perfect model).

### 7. **Plotting the Regression Line**
```python
m, b = np.polyfit(X_train['Age'], y_train, 1)
plt.scatter(X_train['Age'], y_train)
plt.plot(X_train['Age'], m*X_train['Age'] + b, color='red')
plt.show()
```
- **np.polyfit(X_train['Age'], y_train, 1)**: Fits a straight line (degree 1) to the training data, returning the slope (`m`) and intercept (`b`) of the line.
- **plt.scatter(X_train['Age'], y_train)**: Creates a scatter plot of the training data points, where `Age` is on the x-axis and `UsageDuraiton` is on the y-axis.
- **plt.plot(X_train['Age'], m*X_train['Age'] + b, color='red')**: Plots the regression line using the slope and intercept from the polyfit function.
- **plt.show()**: Displays the scatter plot with the regression line. This helps visualize how well the model fits the data.
