# EXP-10 Data Science Process on Complex Dataset

# AIM
To Perform Data Science Process on a complex dataset and save the data to a file.

# ALGORITHM

## Step 1

Read the given Data

## Step 2

Clean the Data Set using Data Cleaning Process

## Step 3

Apply Feature Generation/Feature Selection Techniques on the data set

## Step 4

Apply EDA /Data visualization techniques to all the features of the data set

# CODE
```
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

df.head()

df.isnull().sum()

plt.figure(figsize=(5,5))

plt.title("Data with Outliers")

df.boxplot()

plt.show()

plt.figure(figsize=(5,5))

cols = ['size','tip','total_bill']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

df['sex'].unique()

!pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

be = BinaryEncoder()

data = be.fit_transform(df['sex'])

df = pd.concat([df,data],axis=1)

df

df['smoker'].unique()

data = be.fit_transform(df['smoker'])

df = pd.concat([df,data],axis=1)

df

df['day'].unique()

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

clim = ['Thur','Fri','Sat','Sun']

en= OrdinalEncoder(categories = [clim])

df['day']=en.fit_transform(df[["day"]])

df

df['time'].unique()

le = LabelEncoder()

df['time'] = le.fit_transform(df[["time"]])

df

df.drop('sex',axis=1,inplace=True)

df.drop('smoker',axis=1,inplace=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("Min-max scaled data:")

print(scaled_data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print("Standard scaled data:")

print(scaled_data)

import seaborn as sns

sns.scatterplot(data=df)

sns.displot(df['size'],kde=True)

sns.scatterplot(x="total_bill", y="tip", data=df)

plt.title("Correlation between Tip Amount and Total Bill Amount")

plt.show()

df["tip_percent"] = df["tip"] / df["total_bill"]

sns.barplot(x=df['size'],y=df['tip_percent'],data=df)

plt.title("Tip Percentage by Dining Party Size")

plt.show()

sns.barplot(x=df['time'], y=df['total_bill'])

plt.title("Highest Total Bill Amount by Time")

plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)
```
# OUTPUT
![image](https://github.com/swathidd/EX-10-/assets/121300272/1c04ad11-02e3-4142-a3f7-49602ca5ca57)

![image](https://github.com/swathidd/EX-10-/assets/121300272/684f699c-0746-436c-9682-622bb78ed414)

![image](https://github.com/swathidd/EX-10-/assets/121300272/af2ffc46-8065-485b-aeee-1bd6b4e50ad4)

![image](https://github.com/swathidd/EX-10-/assets/121300272/101fa5eb-33c9-4c19-881a-f0d17af0aca6)

![image](https://github.com/swathidd/EX-10-/assets/121300272/a72013ba-068c-49d5-8436-b289f54a4af2)

![image](https://github.com/swathidd/EX-10-/assets/121300272/4218fea9-1d15-4f00-ab15-d8f7981b7242)

![image](https://github.com/swathidd/EX-10-/assets/121300272/1994e9f3-a0b6-484c-b77f-beab9cd84ca8)

![image](https://github.com/swathidd/EX-10-/assets/121300272/9764ca84-3432-480a-82a8-757fea05e09e)

![image](https://github.com/swathidd/EX-10-/assets/121300272/dfb9b092-e5c9-40c0-aa8f-146167ae49cf)

![image](https://github.com/swathidd/EX-10-/assets/121300272/b0c499c2-8e45-4587-be15-90632181a811)

![image](https://github.com/swathidd/EX-10-/assets/121300272/a19358c9-5fd9-4b5b-b208-fc04545aa197)

![image](https://github.com/swathidd/EX-10-/assets/121300272/afbd9f0d-3d44-4e0d-8910-2f69f9ae2d1b)

![image](https://github.com/swathidd/EX-10-/assets/121300272/ebe75a80-017e-4112-a74f-5c45a8f3fa09)

![image](https://github.com/swathidd/EX-10-/assets/121300272/8f79d54c-09c7-4eeb-abd9-f16b7daeb7e1)

![image](https://github.com/swathidd/EX-10-/assets/121300272/ed56cfc9-2095-41dc-80d9-6f750bcf54a7)

# RESULT
Thus Data Science Process on a complex dataset was performed successfully.


