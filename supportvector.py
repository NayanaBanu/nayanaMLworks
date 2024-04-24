import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.metrics import accuracy_score
data_set= pd.read_csv(r'surveylungcancer.csv')
df=pd.DataFrame(data_set)
print(df.to_string())
x= data_set.iloc[:, [2,3]].values
y= data_set.iloc[:, 4].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
print("X-train After transform")
df2=pd.DataFrame(x_train)
print(df2.to_string())
print("X-test After transform")
df3=pd.DataFrame(x_test)
print(df3.to_string())
from sklearn.svm import SVC # "Support vector classifier"
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)
y_pred= classifier.predict(x_test)
df2=pd.DataFrame({"Actual Y_Test":y_test,"Prediction Data":y_pred})
print("prediction status")
print(df2.to_string())
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))
test=[[30,2]]
test= st_x.transform(test)
df7=pd.DataFrame(test)
print(df7.to_string())
y_pred_2= classifier.predict(test)
print(y_pred_2)