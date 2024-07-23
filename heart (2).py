def heart():
  import numpy as np
  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score
  # loading the csv data to a Pandas DataFrame
  heart_data = pd.read_csv("/Users/vishal/Desktop/heart.csv")
  # print first 5 rows of the dataset
  heart_data.head()
  # print last 5 rows of the dataset
  heart_data.tail()
  # number of rows and columns in the dataset
  heart_data.shape
  # getting some info about the data
  heart_data.info()
  # checking for missing values
  heart_data.isnull().sum()
  # statistical measures about the data
  heart_data.describe()
  heart_data['target'].value_counts()
  X = heart_data.drop(columns='target', axis=1)
  Y = heart_data['target']
  print()
  print(X)
  print()
  print(Y)
  print()
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
  print(X.shape, X_train.shape, X_test.shape)
  print()
  model = LogisticRegression(solver='lbfgs',max_iter=10000)
  # training the LogisticRegression model with Training data
  model.fit(X_train.values, Y_train.values)
  # accuracy on training data
  X_train_prediction = model.predict(X_train.values)
  training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
  print('Accuracy on Training data : ', training_data_accuracy)
  print()
  # accuracy on test data
  X_test_prediction = model.predict(X_test.values)
  test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
  print('Accuracy on Test data : ', test_data_accuracy)
  print()
  age=int(input("Enter age : "))
  print()
  print("""Enter Gender)
  1 for Male
  0 for Female""")
  sex=int(input("Enter 1 or 2 : "))
  print()
  print("""Enter chest pain type
  1 for typical type 1 angina pain
  2 for typical type angina pain
  3 for non-angina pain
  4 for asymptomatic pain""")
  cp=int(input("Enter 1,2,3 or 4 : "))
  print()
  print("Enter Trest Blood Pressure")
  trestbps=int(input("mm Hg on admission to the hospital : "))
  print()
  chol=int(input("Enter Serum Cholestrol level : "))
  print()
  print("""Enter Fasting Blood Suger
  1 for > 120 mg/dl
  0 for < 120 mg/dl""")
  fbs=int(input("Enter 1 or 0 : "))
  print()
  restecg=int(input("Enter ECG value : "))
  print()
  thalach=int(input("Enter maximum heart achieved : "))
  print()
  print("""If there is any exercise induced angina
  1 for Yes
  0 fro No""")
  exang=int(input("Enter 1 or 0 : "))
  print()
  print("""Enter ST depression induced by excercise relative to rest""")
  oldpeak=float(input("Enter value between 0 to 6.2 : "))
  print()
  print("""Enter the slope of the peak exercise ST segment
  1 for Upsloping
  2 for Flat
  3 for Downsloping""")
  slope=int(input("Enter 1,2 or 3 : "))
  print()
  ca=int(input("Number of major vessels colored by fluoroscopy : "))
  print()
  print("""Enter Thalassemia level
  1 for normal
  2 for fixed defect
  3 for reversible defect""")
  thal=int(input("Enter 1,2 or 3 : "))
  print()

  input_data = (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)

  # change the input data to a numpy array
  input_data_as_numpy_array = np.asarray(input_data)

  # reshape the numpy array as we are predicting for only on instance
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

  prediction = model.predict(input_data_reshaped)
  if (prediction[0]== 0):
    print('The Person does not have a Heart Disease')
  else:
    print('The Person has Heart Disease')
heart()
