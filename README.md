# Semenar-DataScienceAndBigData : Chapter-1 Practice before joining the seminar

## LAB1-Introduction to Python on Google Colab
- 1.1 print("Hello STOU")

```py
print("Hello STOU")
```
<br>

- 1.2 Single line comment

```py
print("Hello STOU") #first command
```
<br>

- 1.3 Multiline comments

```py
x = 5
y = "Google"
z = "Colab"

'''
print(x)
print(y)
print(z)
'''
```
<br>

- 1.4 Python basic operators

```py
num1 = 10 # int
num2 = 5.2 # float

num3 = num1+num2

print(num3) # 15.2
```
<br>

- 1.5 Python concatenate strings

```py
firstname="URAI"
lastname="PAITOON"

fullname = firstname+lastname

print(fullname) # URAIPAITOON
```
<br>

- 1.6 Modules

```py
import pandas as pd #all files
import matplotlib.pyplot as plt
import numpy as np
```
<br>

- 1.7 Read File

```py
data = pd.read_csv('/content/sample_data/gradingsystem_training.csv')
print(data)
```

```shell
    StudentNo  Science  Math  CGPA
0           1       22    27   2.5
1           2       23    17   2.3
2           3       31    25   2.5
3           4       27    22   2.4
4           5       32    20   2.3
5           6       32    20   2.3
6           7       23    15   2.2
7           8       18    20   2.8
8           9       26    22   2.8
9          10       30    17   2.5
10         11       30    20   2.3
11         12       30    15   2.4
12         13       30    27   2.8
13         14       30    22   3.0
14         15       30    20   3.0
15         16       62    25   3.0
16         17       50    17   2.0
17         18       50    20   2.3
18         19       50    12   2.0
19         20       50    20   2.3
20         21       50    15   2.2
21         22       58    60   4.0
22         23       33    20   2.3
23         24       58    62   4.0
24         25       40    22   2.8
25         26       40    15   2.2
26         27       40    20   2.8
27         28       40    17   2.5
28         29       40    12   2.0
29         30       44    40   3.8
```

- 1.8 count all rows

```py
count_data = len(data)
print("Count all rows:" + str(count_data)) # Count all rows:30
```
<br>

- 1.9 Plot Graph

```py
x = data['Math']
y = data['CGPA']

plt.scatter(x,y,color='red')
plt.xlabel('Math', fontsize=14)
plt.ylabel('CGPA', fontsize=14)

plt.show()
```

![01](/01.png)
<br>

## LAB2- Python Libraries for Data Science (votebyprovince.csv)

```py
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('/content/sample_data/votebyprovince.csv')

print(data)
```

```shell
     province  vote
0  Chiang Mai  2900
1     Bangkok  2360
2       Trang   190
3      Phuket  2400
4   Ayutthaya  1590
5       Krabi  2150
```
<br>


```py
data.sort_values(by=['vote'],ascending=True)
```

```shell

    province	vote
2	Trang	    190
4	Ayutthaya	1590
5	Krabi	    2150
1	Bangkok	    2360
3	Phuket	    2400
0	Chiang Mai	2900
```
<br>

- Plot Graph

```py
# Define the data for plotting
x = data['province']
y = data['vote']

# Plot the bar graph
plt.bar(x, y, color='green')
plt.xlabel('Province', fontsize=14)
plt.ylabel('Vote', fontsize=14)
plt.title('Popular vote by province', fontsize=14)

# Show the plot
plt.show()
```
![02](/02.png)
<br>

- Plot Graph with label
```py
# Plot Graph with label
def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i])

x = data['province']
y = data['vote']
plt.bar(x, y, color='green')

addlabels(x, y)

plt.xlabel('Province', fontsize=14)
plt.ylabel('Vote', fontsize=14)
plt.title('Popular vote by province', fontsize=14)
plt.show()
```

![03](/03.png)

## LAB3- grading system (GradingSystem.csv)

```py
# ทดสอบคะแนนคณิตศาสตร์ = 30
X = 30  # คะแนนคณิตศาสตร์

def predictCGPAScore():
    a = 1.64536225  # จุดตัด
    b = 0.04122299  # สัมประสิทธิ์ถดถอย
    error = 0
    y = a + np.sum(b * X) + 0
    print(y)

predictCGPAScore() # 2.88205195
```
<br>

```shell
pip install scikit-learn
```
<br>

```py
# การนำเข้าโมดูลต่างๆ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn # ติดตั้ง sklearn ด้วยคำสั่ง pip install scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# read data
pd.set_option('display.max_rows', None)
dataset = pd.read_csv('/content/sample_data/gradingsystem_training.csv')
print(dataset)
```

```shell
    StudentNo  Science  Math  CGPA
0           1       22    27   2.5
1           2       23    17   2.3
2           3       31    25   2.5
3           4       27    22   2.4
4           5       32    20   2.3
5           6       32    20   2.3
6           7       23    15   2.2
7           8       18    20   2.8
8           9       26    22   2.8
9          10       30    17   2.5
10         11       30    20   2.3
11         12       30    15   2.4
12         13       30    27   2.8
13         14       30    22   3.0
14         15       30    20   3.0
15         16       62    25   3.0
16         17       50    17   2.0
17         18       50    20   2.3
18         19       50    12   2.0
19         20       50    20   2.3
20         21       50    15   2.2
21         22       58    60   4.0
22         23       33    20   2.3
23         24       58    62   4.0
24         25       40    22   2.8
25         26       40    15   2.2
26         27       40    20   2.8
27         28       40    17   2.5
28         29       40    12   2.0
29         30       44    40   3.8
```
<br>

```py
# set data x, y
x = dataset['Math']
y = dataset['CGPA']

# plot graph
plt.scatter(x, y, color='red')
plt.xlabel('MATH', fontsize=14)
plt.ylabel('CGPA', fontsize=14)
plt.show()
```
![04](/04.png)
<br>

```py
#กำหนดค่า x,y
x = dataset.iloc[:, 2].values.reshape(-1, 1) #อาร์เรย์ตัวแปรอิสระ
y = dataset.iloc[:, 3].values.reshape(-1, 1) #อาร์เรย์ตัวแปรตาม

#การแบ่งข้อมูลออกเป็น 70:30
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3,random_state=0)

#การสร้างแบบจำลองการถดถอยเชิงเส้นอย่างง่าย
regression_model = LinearRegression()
regression_model.fit(x_train,y_train)

#การทำนายข้อมูล
y_predicted = regression_model.predict(x_test)
y_predicted

#การแสดงค่าจริง และค่าทำนาย
df = pd.DataFrame({'Actual': [y_test], 'Predicted': [y_predicted]})
print(df)

#การวัดประสิทธิภาพของแบบจำลองการถดถอยเชิงเส้นอย่างง่าย
rmse = mean_squared_error(y_test, y_predicted)
r2_score = r2_score(y_test,y_predicted)

#การแสดงค่าผลการวัดประสิทธิภาพของแบบจำลอง
print('The intercept is:', regression_model.intercept_)
print('The coefficient is:' , regression_model.coef_)
print('The rmse is:', rmse)
print('The r2_score is:', r2_score)
```

```shell
                                              Actual  \
0  [[2.5], [2.0], [3.0], [2.3], [2.8], [2.8], [2....   

                                           Predicted  
0  [[2.675936945401399], [2.1400381041862406], [2...  
The intercept is: [1.64536225]
The coefficient is: [[0.04122299]]
The rmse is: 0.05792675297327944
The r2_score is: 0.32195563716248055
```
<br>

```py
# ทดสอบทำนายคะแนนคณิตศาสตร์ = 30
X = 30  # คะแนนคณิตศาสตร์

def predictCGPAScore():
    a = 1.64536225  # จุดตัด
    b = 0.04122299  # สัมประสิทธิ์ถดถอย
    error = 0
    y = a + np.sum(b * X) + 0
    print(y)

predictCGPAScore() # 2.88205195
```
<br>

```py
# ทดสอบทำนายคะแนนคณิตศาสตร์ = 20
X = 20  # คะแนนคณิตศาสตร์

def predictCGPAScore():
    a = 1.64536225  # จุดตัด
    b = 0.04122299  # สัมประสิทธิ์ถดถอย
    error = 0
    y = a + np.sum(b * X) + 0
    print(y)

predictCGPAScore() # 2.4698220500000003
```

## LAB4-student dropout (Student.csv)

```py
#การนำเข้าโมดูลต่างๆ
import pandas as pd  # data processing
import numpy as np  # linear algebra
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # statistical data visualization
import sklearn as sk  # machine learning model
import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
```

```python
#อ่านไฟล์ Student.csv
studentdata = pd.read_csv('/content/sample_data/Student.csv')
studentdata.head(5)
```

```shell
	StudentYear	HighSchoolGrade	Age	    Familymember	Gender	CGPA	Address	CoreCourseScore	ElectiveCourseScore	Result
0	        1	            3.0  18	            3.0	    M	    2.5	    Central	    1	            1	                0
1	        1	            3.0  18	            5.0	    F	    2.3	    Central	    1	            1	                0
2	        3	            3.0	 20	            6.0	    F	    2.5	    Central	    1	            1	                0
3	        1	            3.0	 18	            2.0	    F	    2.4	    Central	    1	            1	                0
4	        1	            3.0	 18	            3.0	    M	    2.3	    Southern	1	            1	                0
```
<br>

```py
# การหาคุณลักษณะสำคัญ (feature selection)
all_features = [name for name in studentdata.columns if studentdata[name].dtype == 'object']
all_features

# แปลง gender และ address ให้เป็นตัวเลข
all_features = [name for name in studentdata.columns if studentdata[name].dtype == 'object']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in list(all_features):
    studentdata[i] = le.fit_transform(studentdata[i])
for x in all_features:
    print(x, "=", studentdata[x].unique())
```

```shell
Gender = [1 0]
Address = [0 3 2 4 1]
```
<br>

```py
#แสดงข้อมูลหลังเปลี่ยนเป็นตัวเลข
studentdata.head(5)
```

```shell
	StudentYear	HighSchoolGrade	Age	Familymember	Gender	CGPA	Address	CoreCourseScore	ElectiveCourseScore	Result
0	        1	            3.0	18	3.0	                1	2.5	        0	            1	                1	    0
1	        1	            3.0	18	5.0	                0	2.3	        0	            1	                1	    0
2	        3	            3.0	20	6.0	                0	2.5	        0	            1	                1	    0
3	        1	            3.0	18	2.0	                0	2.4	        0	            1	                1	    0
4	        1	            3.0	18	3.0	                1	2.3	        3	            1	                1	    0
```
<br>

```py
# หาค่า chi2
from sklearn.feature_selection import chi2
studentdata.fillna(0, inplace=True)
X = studentdata.drop('Result', axis=1)
y = studentdata['Result']
chi_scores = chi2(X, y)
chi_scores

# แสดงคุณลักษณะและข้อมูลที่สัมพันธ์กับผลลัพธ์
p_values = pd.Series(chi_scores[1], index=X.columns)
p_values.sort_values(ascending=True, inplace=True)
p_values
```

```shell
CGPA	                0.001333
Gender	                0.004478
ElectiveCourseScore	    0.005175
StudentYear	            0.398843
Address	                0.453260
CoreCourseScore	        0.469059
Age	                    0.574372
Familymember	        0.773231
HighSchoolGrade	        0.785089

dtype: float64
```
<br>

```py
#สร้างกราฟแสดงคุณลักษณะข้อมูลที่สัมพันธ์กับผลลัพธ์
p_values.plot.bar(figsize = (10,5), cmap="coolwarm")
plt.title('Chi-square test for feature selection', size=18) # Text(0.5, 1.0, 'Chi-square test for feature selection')
```

![05](/05.png)
<br>

```py
# นำข้อมูลที่ไม่เกี่ยวข้องกับการสร้างแบบจำลองออก
newfeature = studentdata.columns.tolist()
newfeature.remove('CGPA')
newfeature.remove('Gender')
newfeature.remove('ElectiveCourseScore')
newfeature
```

```shell
['StudentYear',
 'HighSchoolGrade',
 'Age',
 'Familymember',
 'Address',
 'CoreCourseScore',
 'Result']
```
<br>

```py
#สร้าง Algorithm ชื่อว่า Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# การแบ่งชุดข้อมูลนักเรียนออกเป็น 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#การสร้างแบบจำลองต้นไม้ตัดสินใจ
modelDT = DecisionTreeClassifier(random_state=43)

# การฝึกสอนข้อมูล
modelDT.fit(X_train, y_train)

# การทำนายข้อมูล
predictions = modelDT.predict(X_test)

#การสร้างคอนฟิวชันเมทริกซ์สำหรับวัดประสิทธิภาพแบบจำลอง
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
```

```shell
[[41  0]
 [ 2  7]]
              precision    recall  f1-score   support

           0       0.95      1.00      0.98        41
           1       1.00      0.78      0.88         9

    accuracy                           0.96        50
   macro avg       0.98      0.89      0.93        50
weighted avg       0.96      0.96      0.96        50

0.96
```
<br>

---