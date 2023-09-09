import numpy as np
import pickle
 
loaded_model = pickle.load(open("C:/Users/NOCAY/Desktop/DATA SCIENCE/trained_model.sav", 'rb'))

input_data = (1,106,76,0,0,37.5,0.197,26)
#Changing the inputn data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#standardize the input data
#std_data = scaler.transform(input_data_reshaped)
#print(std_data)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')