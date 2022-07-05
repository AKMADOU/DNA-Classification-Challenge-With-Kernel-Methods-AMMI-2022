import pandas as pd
from kernel_logistic_regression import*


Xtr = pd.read_csv('Data/Xtr.csv')
Xtr_vec = pd.read_csv('Data/Xtr_vectors.csv')
Xte = pd.read_csv('Data/Xte.csv')
Xte_vec = pd.read_csv('Data/Xte_vectors.csv')
Ytr= pd.read_csv('Data/Ytr.csv')
Ytr['Covid'] = 2*Ytr['Covid']-1



# First drop the "Id" column
Xtr_vec=Xtr_vec.drop(['Id'],axis=1)

#drop the "Id" columns from the test set
Xte_vec=Xte_vec.drop(['Id'],axis=1) 

# Split the dataset into training set and validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    Xtr_vec, Ytr['Covid'], test_size=0.33, random_state=42)



kernel = 'rbf'
sigma = .46
lambd = .0001
degree = 3
intercept = False

kernel_parameters = {
    'degree': 2,
    'sigma': 0.46,
}
lambd = 0.0001

training_parameters = {
    'fit_intercept': False,
    'lr': 0.0001,
    'method': 'newton'
}

print()
print("################# Training start... ####################")
klr_model = KernelLogisticRegression(lambd=lambd, kernel=kernel, **kernel_parameters)

klr_model.fit(X_train.to_numpy(), y_train.to_numpy(), **training_parameters)
# print('train', X_train.shape)
y_pred_train=klr_model.predict(X_train.to_numpy())
y_pred = klr_model.predict(X_test.to_numpy())
print("Validation score : ", accuracy_score(y_pred,y_test))


#Predictions
pred=klr_model.predict(Xte_vec.to_numpy())
pred = pd.DataFrame(pred)
pred.columns=['Covid']
pred['Id']=Xte['Id']
prediction = pd.DataFrame()
prediction['Id']=pred['Id']
prediction['Covid']=pred['Covid']

# Change -1 to 0
prediction[prediction==-1]=0
prediction['Covid'] = prediction['Covid'].apply(lambda x: int(x))

#Submission
prediction.to_csv('Yte.csv',index=False)
print('######## Submission file generate successfully ##################')
print()

