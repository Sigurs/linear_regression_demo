import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_csv('challenge_dataset.txt', names=['x','y'])
#print(dataframe.x)
#exit()
x_values = dataframe['x'].to_frame()
y_values = dataframe['y'].to_frame()

#print(dataframe.keys())
#print(dataframe['x'])
#print(dataframe['x'].values)
print(x_values)
print(y_values)

print(len(x_values))
print(len(y_values))
print(type(y_values))

#exit()
#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# 95 13.3940  9.05510
print('13.3940')
print(body_reg.predict(13.3940))
#visualize results

plt.scatter(x_values, y_values)
#plt.plot(x_values, body_reg.predict(x_values))
plt.plot([0,13.3940], [0,body_reg.predict(13.3940)])
plt.show()
