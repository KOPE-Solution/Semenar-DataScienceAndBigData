# Semenar-DataScienceAndBigData : Chapter-3 กิจกรรมเดี๋ยว

## A predictive Analytics Model for Students Grade

- 2.1 Import libraries
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn #ติดตั้ง sklearn ด้วยคำสั่ง pip install scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```
<br>

- 2.2 Read file
```py
dataset = pd.read_csv('/content/sample_data/gradingsystem_training.csv')
dataset.head()
```

```shell
	StudentNo	Science	Math	CGPA
0	1	            22	27	    2.5
1	2	            23	17	    2.3
2	3	            31	25	    2.5
3	4	            27	22	    2.4
4	5	            32	20	    2.3
```
<br>

- 2.3 Display 2 variables with graph
```py
# Define x and y from the dataset
x = dataset['Math']  # Assuming 'Math' is the column name for math scores
y = dataset['CGPA']  # Assuming 'CGPA' is the column name for CGPA scores

# Plot the scatter plot
plt.scatter(x, y, color='red')
plt.xlabel('MATH', fontsize=14)
plt.ylabel('CGPA', fontsize=14)
plt.show()
```

![01](/01.png)
<br>

- 2.4 Set x and y variable for generating a model
```py
x = dataset.iloc[:, 2].values.reshape(-1, 1) #อาร์เรย์ตัวแปรอิสระ
y = dataset.iloc[:, 3].values.reshape(-1, 1) #อาร์เรย์ตัวแปรตาม
```
<br>

- 2.5 Splitting data for machine learning model (70:30)
```py
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0)
```
<br>

2.6 Building a Linear Regression Model
```py
regression_model = LinearRegression()
regression_model.fit(x_train,y_train)
```
<br>

- 2.7 Prediction
```py
y_predicted = regression_model.predict(x_test)
y_predicted
y_test

df = pd.DataFrame({'Actual': [y_test], 'Predicted': [y_predicted]})
print(df)
```

```shell
                                              Actual  \
0  [[2.5], [2.0], [3.0], [2.3], [2.8], [2.8], [2....   

                                           Predicted  
0  [[2.675936945401399], [2.1400381041862406], [2...  
```
<br>

- 2.8 Model Evaluation
```py
rmse = mean_squared_error(y_test, y_predicted)
r2_score = r2_score(y_test,y_predicted)
```

```py
print('Evaluation Result:\n--------------------------------')
print('The intercept is:', regression_model.intercept_)
print('The coefficient is:', regression_model.coef_)
print('The RMSE is:', rmse)
print('The R^2 score is:', r2_score)
```

```shell
Evaluation Result:
--------------------------------
The intercept is: [1.64536225]
The coefficient is: [[0.04122299]]
The RMSE is: 0.05792675297327944
The R^2 score is: 0.32195563716248055
```
<br>

- Q1: If he/she get a math score = 30 then What a CGPA of this student?
```py
# Predict CGPA for a math score of 30
math_score = np.array([[30]])  # Math score needs to be in 2D array format
predicted_cgpa = regression_model.predict(math_score)
print(f"Predicted CGPA for a math score of 30: {predicted_cgpa[0][0]}")
```

```shell
Predicted CGPA for a math score of 30: 2.882051884330306
```

---