import numpy as np
import pickle

#loading the save model
loaded_model = pickle.load(open(r'C:\Users\HP\OneDrive\Desktop\Diabetes_prediction\Trained_model.sav'
 , 'rb')) #rb --> read binary

input_data = (5,166,72,19, 175, 25.8 , 0.587 , 51)

#changing the input data into the numpy array

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) #reshaping the data.

prediction = loaded_model.predict(input_data_reshaped)

print(prediction)

if(prediction[0] ==0):
  print("The person is not diabetic")
else:
  print("The person is Diabetic")