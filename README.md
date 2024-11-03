# Semenar-DataScienceAndBigData : Chapter-2 แบบฝึกปฏิบัติสัมนาเสริม

## 1) กรณีศึกษาการวิเคราะห์ราคาของสังหาริมทรัพย์โดยใช้การถดถอยเชิงเส้นพหุคูณ

```shell
ที่มาโจทย์ปัญหาวิเคราะห์ราคาคอนโดมิเนียมโดยใช้การถดถอยเชิงเส้นพหุคูณ เนื่องจากการกำหนดราคาของสังหาริมทรัพย์ โดยเฉพาะราคาคอนโดมิเนียม เกี่ยวข้องกับปัจจัยต่างๆ มากมาย เช่น ขนาดห้อง จำนวนห้องนอน ห้องน้ำ สถานที่ หรือทำเล ชั้นของห้องพัก และปัจจัยอื่นๆ ด้วย ซึ่งปัจจัยต่างๆ เหล่านี้มีผลต่อการกำหนดราคาคอนโดมิเนียมที่ถูกต้อง เมื่อกำหนดราคาคอนโดมิเนียมที่ถูกต้อง จะส่งเสริมการซื้อขาย และลงทุน มีผลต่อความเจริญของธุรกิจสังหาริมทรัพย์ ดังนั้นกรณีศึกษานี้ทำการศึกษาการทำนายหรือวิเคราะห์ราคาคอนโดมิเนียมโดยใช้การถดถอยเชิงเส้นพหุคูณ สำหรับการกำหนดราคาที่ถูกต้อง โดยการนำตัวแปรต่างๆ มาวิเคราะห์ว่าอิทธิพลของตัวแปรใดที่มีผลต่อการทำนายราคาคอนโดมิเนียมที่ถูกต้อง โดยตัวแปรต่างๆ ที่ใช้ในการศึกษามีทั้งตัวแปรอิสระ (independence variable) และตัวแปรตาม (dependence variable) ตัวแปรอิสระ (independence variable) เช่น แกน (X) ส่วนตัวแปรตาม (dependence variable) หรือ แกน (Y) สำหรับสร้างแบบจำลองการทำนายราคาคอนโดมิเนียม ที่ถูกต้องและเหมาะสม
```

```shell
ขั้นตอนการวิเคราะห์ราคาคอนโดมิเนียมโดยใช้การถดถอยเชิงเส้นพหุคูณด้วยไพธอน ประกอบด้วย 3 กระบวนการทำงานหลัก ได้แก่การนำข้อมูลเข้า (input data) การประมวลผลข้อมูล (data processing) และการแสดงผลลัพธ์ (output)
```

![02](/02.png)
<br>

## ขั้นตอนการเขียนคำสั่ง รายละเอียดดังต่อไปนี้
1) นำเข้า library และแสดงข้อมูล

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
mycondo = pd.read_csv('/content/sample_data/Condo.csv')
mycondo.head(5)
```

```shell
	bedrooms	bathrooms	sqft_lot	floors	zone	price
0	        3	        1	    1180	1.0	    sathon	221900
1	        3	        2	    2570	2.0	    sathon	538000
2	        2	        1	    770	    1.0	    sathon	180000
3	        4	        3	    1960	1.0	    sathon	604000
4	        3	        2	    1680	1.0	    sathon	510000
```
<br>

2) ตรวจสอบความสัมพันธ์ระหว่างข้อมูลกับคลาสเป้าหมาย
```py
chkfeatures = [name for name in mycondo.columns if mycondo[name].dtype == 'object']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in list(chkfeatures):
    mycondo[i] = le.fit_transform(mycondo[i])
    
for x in chkfeatures:
    print(x, "= ", mycondo[x].unique())

# zone =  [3 0 1 2]
```
<br>

3) แสดงข้อมูล 10 รายการ
```py
mycondo.head(10)
```

```shell
	bedrooms	bathrooms	sqft_lot	floors	zone	price
0	        3	        1	    1180	1.0	        3	221900
1	        3	        2	    2570	2.0	        3	538000
2	        2	        1	    770	    1.0	        3	180000
3	        4	        3	    1960	1.0	        3	604000
4	        3	        2	    1680	1.0	        3	510000
5	        4	        3	    5420	1.0	        3	1225000
6	        3	        2	    1715	2.0	        3	257500
7	        3	        1	    1060	1.0	        3	291850
8	        3	        1	    1780	1.0	        3	229500
9	        3	        2	    1890	2.0	        3	323000
```
<br>

4) กราฟแสดงความสัมพันธ์ระหว่างข้อมูล
```py
x = mycondo['zone']
y = mycondo['price']
plt.scatter(x,y,color='red')
plt.xlabel('zone', fontsize=14)
plt.ylabel('price', fontsize=14)
plt.show()
```

![01](/01.png)
<br>

5) สร้างแบบจำลองด้วย Linear Regression
```py
from sklearn.linear_model import LinearRegression

x = mycondo[['bedrooms', 'zone']]
y = mycondo['price']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, test_size=0.2,random_state=0)
modelRE = LinearRegression()
modelRE.fit(x_train,y_train)
```
<br>

6) การหาสัมประสิทธิ์ความสัมพันธ์ระหว่างข้อมูล

```py
coeff1 = pd.DataFrame(modelRE.coef_, x.columns, columns=['Coefficient'])
coeff1
```

```shell
	        Coefficient
bedrooms	93951.238626
zone	    -10152.434094
```
<br>

7) การหาค่าข้อมูลจริง และข้อมูลการทำนาย
```py
y_predicted = modelRE.predict(x_test)
y_predicted
y_test
df = pd.DataFrame({'Actual': [y_test], 'Predicted': [y_predicted]})
print(df)
```

```shell
                                              Actual  \
0  18      189000
170     284000
107     188500
9...   

                                           Predicted  
0  [368654.5375249464, 482910.6443383652, 378806....  
```
<br>

8) การแสดงค่า a, b1 และ b2
```py
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, y_predicted)
print('Evaluation Result :\n--------------------------------')
print('The intercept is:', modelRE.intercept_)
print('The coefficient is:' , modelRE.coef_)
print('The rmse is:',rmse)
```

```shell
Evaluation Result :
--------------------------------
The intercept is: 211209.36255376774
The coefficient is: [ 93951.23862609 -10152.43409367]
The rmse is: 63924545015.29665
```
<br>

9) การทำนายราคาคอนโด
```py
# Example values (replace these with the actual model intercept and coefficients)
a = modelRE.intercept_
b1, b2 = modelRE.coef_

# Predicted feature values
X1 = 3  # For example, bedrooms
X2 = 2  # For example, zone

# Prediction function
def predictCondoPrice():
    Y = a + (b1 * X1) + (b2 * X2)
    print("Predicted Condo Price:", Y)

# Call the prediction function
predictCondoPrice() # Predicted Condo Price: 472758.2102446996
```

---