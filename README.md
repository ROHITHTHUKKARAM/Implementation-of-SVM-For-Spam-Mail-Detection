# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1: Start

STEP 2: Detect Encoding & Load Data

STEP 3: Prepare Features and Labels

STEP 4: Split Data into Training and Test Sets

STEP 5: Vectorize Text Data

STEP 6: Train SVM Model and Evaluate

STEP 7: END

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Rohith T
RegisterNumber: 212223040173
*/
```
```
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
## Data.head():
![image](https://github.com/user-attachments/assets/92d26bbb-610e-4492-be2b-2f696009c231)
## data.info():
![image](https://github.com/user-attachments/assets/6a4e1395-3380-459a-8ac1-a957c3dacb38)
## y_pred:
![image](https://github.com/user-attachments/assets/bf70d639-6dd5-4869-b6ca-2559929b28f4)
## accuracy:
![image](https://github.com/user-attachments/assets/4f55b22e-b847-4d0a-a79a-c817caf9c6a1)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
