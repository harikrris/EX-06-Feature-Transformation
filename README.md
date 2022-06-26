# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

df=pd.read_csv("Data_To_Transform.csv")
df

df.skew()

np.log(df["Highly Positive Skew"])

np.reciprocal(df["Moderate Positive Skew"])

np.sqrt(df["Highly Positive Skew"])

np.square(df["Highly Negative Skew"])

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df

df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])
df

df.skew()

from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()

df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])
sm.qqplot(df['Moderate Positive Skew'],line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew_1'],line='45')
plt.show()

df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])
sm.qqplot(df['Highly Positive Skew'],line='45')
plt.show()

sm.qqplot(df['Highly Positive Skew_1'],line='45')
plt.show()

df

# OUPUT
![1](https://user-images.githubusercontent.com/79370364/175799031-5d587d6e-fcad-460d-ae02-f041cb5efb3a.png)
![2](https://user-images.githubusercontent.com/79370364/175799032-4a4a39e0-ccd1-4565-90f4-319e0fd20bfd.png)
![3](https://user-images.githubusercontent.com/79370364/175799033-3f4b9c7a-8f8f-4df5-9e68-0f916a3a170e.png)
![4](https://user-images.githubusercontent.com/79370364/175799034-7fdd134f-de07-4297-a033-9f289309718a.png)
![5](https://user-images.githubusercontent.com/79370364/175799035-b916bcf9-e4f6-49b0-bc3b-19e498a99499.png)
![6](https://user-images.githubusercontent.com/79370364/175799036-cbb27a36-241c-4fef-9ca3-2e2d45ce98e9.png)
![7](https://user-images.githubusercontent.com/79370364/175799038-01bd5c67-4c2e-4e68-bbcf-4ea2392c538d.png)
![8](https://user-images.githubusercontent.com/79370364/175799039-5e1d4c99-697e-45e5-ac0d-92af955bc403.png)
![9](https://user-images.githubusercontent.com/79370364/175799041-6bf40844-1dea-4b22-a1e1-1d672c829e6d.png)
![10](https://user-images.githubusercontent.com/79370364/175799043-792aabce-43f7-4e92-b166-6248b24c6e41.png)
![11](https://user-images.githubusercontent.com/79370364/175799044-9f1e5767-6517-4208-9906-58de573f7ffa.png)
![12](https://user-images.githubusercontent.com/79370364/175799045-863393dd-bf07-4b28-be7a-5494ed3b07eb.png)
![13](https://user-images.githubusercontent.com/79370364/175799046-482f7b3b-2f04-4f48-bbf7-2e27dbfd8941.png)
![14](https://user-images.githubusercontent.com/79370364/175799047-72ac7d18-3259-402a-9c16-281a8a646cc4.png)
![15](https://user-images.githubusercontent.com/79370364/175799048-8ac3030c-1b10-4cbd-ae39-c7564d797016.png)
![16](https://user-images.githubusercontent.com/79370364/175799049-8875a4a9-837e-4c97-b24b-11b3c7995f12.png)
![17](https://user-images.githubusercontent.com/79370364/175799050-ebe1c966-acee-43f9-a792-eb0ff1365e84.png)
![18](https://user-images.githubusercontent.com/79370364/175799051-516c3490-b7e3-4926-8569-747b876d4dab.png)
![19](https://user-images.githubusercontent.com/79370364/175799052-33522803-7368-4f39-9271-01fe1b6586c6.png)
![20](https://user-images.githubusercontent.com/79370364/175799053-acc7fd40-d3ee-4161-b780-ce11eccbc2e1.png)
![21](https://user-images.githubusercontent.com/79370364/175799054-3577c874-2663-4fec-8434-4c44bef9e68a.png)
![22](https://user-images.githubusercontent.com/79370364/175799055-be84a026-85eb-4d45-8a57-d95e05e3ab2f.png)
![23](https://user-images.githubusercontent.com/79370364/175799057-56ddecd7-db0c-4aff-83ce-66b552f09bcc.png)
![24](https://user-images.githubusercontent.com/79370364/175799058-c1e8cd62-147c-4160-8d98-8bd96e20e752.png)
![25](https://user-images.githubusercontent.com/79370364/175799059-e673e81d-86a1-47bb-972a-e20e2f7bd47e.png)
![26](https://user-images.githubusercontent.com/79370364/175799060-614c8779-bbfd-4002-aa80-aa26667e2f9c.png)
![27](https://user-images.githubusercontent.com/79370364/175799061-a9a7962d-349b-4acc-9182-76ee5f509d92.png)
![28](https://user-images.githubusercontent.com/79370364/175799063-e928f861-bd49-421e-8767-2e43a0c6f019.png)
![29](https://user-images.githubusercontent.com/79370364/175799064-ae6b49a4-0798-49ba-a021-b25f24971354.png)
![30](https://user-images.githubusercontent.com/79370364/175799065-187479e0-4129-4b67-a7cd-0a9e616e15cc.png)
![31](https://user-images.githubusercontent.com/79370364/175799066-8a14966a-df97-401d-bc8e-bfe9d2be45aa.png)
![32](https://user-images.githubusercontent.com/79370364/175799067-9e42dc8e-5f81-4d24-940e-ca5926812e73.png)
![33](https://user-images.githubusercontent.com/79370364/175799068-b98e3126-d362-47db-b9c1-84c7781d01f6.png)
![34](https://user-images.githubusercontent.com/79370364/175799069-3e9e70b9-6df2-4245-832d-4064a939074d.png)
![35](https://user-images.githubusercontent.com/79370364/175799070-31f42f8e-ba23-420a-8f15-1b0ed06334c7.png)
![36](https://user-images.githubusercontent.com/79370364/175799072-bc4d5fe7-07eb-4f21-a4ba-d867894c9e75.png)
![37](https://user-images.githubusercontent.com/79370364/175799073-99b9538f-46ca-431f-87a1-1c49a6de9943.png)
![38](https://user-images.githubusercontent.com/79370364/175799074-c65aab73-f748-46d7-a55a-6e8ace52d87a.png)
![39](https://user-images.githubusercontent.com/79370364/175799075-08f92fc9-f9d0-4323-97e8-91a69d2cb38c.png)
