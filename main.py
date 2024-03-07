#gerekli importlar
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

#Read csv
dataset = pd.read_csv("ParisHousing.csv")
#print(dataset)

square_maters = dataset.iloc[:,0].values
number_of_rooms = dataset.iloc[:,1].values
made = dataset.iloc[:,8].values
has_storage_room = dataset.iloc[:,-3].values
has_guest_room = dataset.iloc[:,-2].values
price = dataset.iloc[:,-1].values


#Convert dataframe
square_maters_df = pd.DataFrame(data=square_maters, columns=["square maters"])
number_of_rooms_df = pd.DataFrame(data=number_of_rooms, columns=["Number of Rooms"])
made_df = pd.DataFrame(data=made, columns=["Made"])
storage_room_df = pd.DataFrame(data=has_storage_room, columns=["Has Storage Room"])
guest_room_df = pd.DataFrame(data=has_guest_room, columns=["Has Guest Room"])
price_df = pd.DataFrame(data=price, columns=["Price"])

 
# concat  merge dataframe
df = pd.concat([square_maters_df,number_of_rooms_df,made_df,storage_room_df,guest_room_df,price_df], axis=1)


# Split train and test data
from sklearn.model_selection import train_test_split
#features
X = df.iloc[:,:-1].values
#targets
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)


#Model 
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
# tahmin(predict)
y_pred = reg.predict(X_test)

# metrics
from sklearn.metrics import  mean_squared_error, r2_score

print(f"Price : {mean_squared_error(y_test, y_pred)}")

#R2 hesaplayalım 
r2 = r2_score(y_test, y_pred)
print("R2 Skoru ",r2)


















