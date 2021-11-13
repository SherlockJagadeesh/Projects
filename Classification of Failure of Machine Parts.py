# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Train Dataset
dataset_train_whole = pd.read_csv('C:/Users/HP/Downloads/train_data_iitm.csv')

# Checking for missing values
missing_values = dataset_train_whole.isnull()
for column in missing_values.columns.values.tolist():
    print(column)
    print(missing_values[column].value_counts())
    print("")
    
# Data Exploration
dataset_train_whole.shape
dataset_train_whole.info()
dataset_train_whole.describe()

# Data manipulation
dataset_train_whole['type'].replace(to_replace=['L','M','H'],value=[1,2,3],inplace=True) # assign 1,2,3 values based on their quality

# Splitting into required datasets
dataset_train_1 = dataset_train_whole.iloc[:,:9]
dataset_train_2 = dataset_train_whole.iloc[:,:10]
dataset_train_3 = dataset_train_whole.iloc[:,[0,1,2,3,4,5,6,7,8,10]]
dataset_train_4 = dataset_train_whole.iloc[:,[0,1,2,3,4,5,6,7,8,11]]
dataset_train_5 = dataset_train_whole.iloc[:,[0,1,2,3,4,5,6,7,8,12]]
dataset_train_6 = dataset_train_whole.iloc[:,[0,1,2,3,4,5,6,7,8,13]]

# Data Exploration
target1_count = dataset_train_1.iloc[:,-1].value_counts()
target2_count = dataset_train_2.iloc[:,-1].value_counts()
target3_count = dataset_train_3.iloc[:,-1].value_counts()
target4_count = dataset_train_4.iloc[:,-1].value_counts()
target5_count = dataset_train_5.iloc[:,-1].value_counts()
target6_count = dataset_train_6.iloc[:,-1].value_counts()
print(f'Target counts for all the datasets is \n{target1_count} \n{target2_count} \n{target3_count} \n{target4_count} \n{target5_count} \n{target6_count}')

# Data division
X1 = dataset_train_1.iloc[:,2:-1].values
Y1 = dataset_train_1.iloc[:,-1].values
X2 = dataset_train_2.iloc[:,2:-1].values
Y2 = dataset_train_2.iloc[:,-1].values
X3 = dataset_train_3.iloc[:,2:-1].values
Y3 = dataset_train_3.iloc[:,-1].values
X4 = dataset_train_4.iloc[:,2:-1].values
Y4 = dataset_train_4.iloc[:,-1].values
X5 = dataset_train_5.iloc[:,2:-1].values
Y5 = dataset_train_5.iloc[:,-1].values
X6 = dataset_train_6.iloc[:,2:-1].values
Y6 = dataset_train_6.iloc[:,-1].values

# Exploratory Data Analysis for datasets:

# Dataset - 1:
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset_train_1.iloc[:,2:-1].shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset_train_1.iloc[:,2:-1].columns.values[i - 1])

    vals = np.size(dataset_train_1.iloc[:,2:-1].iloc[:, i - 1].unique())
    
    plt.hist(dataset_train_1.iloc[:,2:-1].iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

sns.set(style="white", font_scale=2)
corr = dataset_train_1.iloc[:,2:-1].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

dataset_train_1.iloc[:,2:-1].corrwith(dataset_train_1.iloc[:,-1]).plot.bar(figsize=(20,10),
                  title = 'Correlation with Reponse variable',
                  fontsize = 15, rot = 45,
                  grid = True)

# Dataset - 2:
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset_train_2.iloc[:,2:-1].shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset_train_2.iloc[:,2:-1].columns.values[i - 1])

    vals = np.size(dataset_train_2.iloc[:,2:-1].iloc[:, i - 1].unique())
    
    plt.hist(dataset_train_2.iloc[:,2:-1].iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

sns.set(style="white", font_scale=2)
corr = dataset_train_2.iloc[:,2:-1].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

dataset_train_2.iloc[:,2:-1].corrwith(dataset_train_2.iloc[:,-1]).plot.bar(figsize=(20,10),
                  title = 'Correlation with Reponse variable',
                  fontsize = 15, rot = 45,
                  grid = True)

# Dataset - 3:
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset_train_3.iloc[:,2:-1].shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset_train_3.iloc[:,2:-1].columns.values[i - 1])

    vals = np.size(dataset_train_3.iloc[:,2:-1].iloc[:, i - 1].unique())
    
    plt.hist(dataset_train_3.iloc[:,2:-1].iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

sns.set(style="white", font_scale=2)
corr = dataset_train_3.iloc[:,2:-1].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

dataset_train_3.iloc[:,2:-1].corrwith(dataset_train_3.iloc[:,-1]).plot.bar(figsize=(20,10),
                  title = 'Correlation with Reponse variable',
                  fontsize = 15, rot = 45,
                  grid = True)

# Dataset - 4:
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset_train_4.iloc[:,2:-1].shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset_train_4.iloc[:,2:-1].columns.values[i - 1])

    vals = np.size(dataset_train_4.iloc[:,2:-1].iloc[:, i - 1].unique())
    
    plt.hist(dataset_train_4.iloc[:,2:-1].iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

sns.set(style="white", font_scale=2)
corr = dataset_train_4.iloc[:,2:-1].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

dataset_train_4.iloc[:,2:-1].corrwith(dataset_train_4.iloc[:,-1]).plot.bar(figsize=(20,10),
                  title = 'Correlation with Reponse variable',
                  fontsize = 15, rot = 45,
                  grid = True)

# Dataset - 5:
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset_train_5.iloc[:,2:-1].shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset_train_5.iloc[:,2:-1].columns.values[i - 1])

    vals = np.size(dataset_train_5.iloc[:,2:-1].iloc[:, i - 1].unique())
    
    plt.hist(dataset_train_5.iloc[:,2:-1].iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

sns.set(style="white", font_scale=2)
corr = dataset_train_5.iloc[:,2:-1].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

dataset_train_5.iloc[:,2:-1].corrwith(dataset_train_5.iloc[:,-1]).plot.bar(figsize=(20,10),
                  title = 'Correlation with Reponse variable',
                  fontsize = 15, rot = 45,
                  grid = True)


# Dataset - 6:
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset_train_6.iloc[:,2:-1].shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset_train_6.iloc[:,2:-1].columns.values[i - 1])

    vals = np.size(dataset_train_6.iloc[:,2:-1].iloc[:, i - 1].unique())
    
    plt.hist(dataset_train_6.iloc[:,2:-1].iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

sns.set(style="white", font_scale=2)
corr = dataset_train_6.iloc[:,2:-1].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

dataset_train_6.iloc[:,2:-1].corrwith(dataset_train_6.iloc[:,-1]).plot.bar(figsize=(20,10),
                  title = 'Correlation with Reponse variable',
                  fontsize = 15, rot = 45,
                  grid = True)

# Train test Split
from sklearn.model_selection import train_test_split
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.10, random_state = 0)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.10, random_state = 0)
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3, Y3, test_size = 0.10, random_state = 0)
X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, Y4, test_size = 0.10, random_state = 0)
X5_train, X5_test, Y5_train, Y5_test = train_test_split(X5, Y5, test_size = 0.10, random_state = 0)
X6_train, X6_test, Y6_train, Y6_test = train_test_split(X6, Y6, test_size = 0.10, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1_train = sc.fit_transform(X1_train)
X1_test = sc.transform(X1_test)
X2_train = sc.fit_transform(X2_train)
X2_test = sc.transform(X2_test)
X3_train = sc.fit_transform(X3_train)
X3_test = sc.transform(X3_test)
X4_train = sc.fit_transform(X4_train)
X4_test = sc.transform(X4_test)
X5_train = sc.fit_transform(X5_train)
X5_test = sc.transform(X5_test)
X6_train = sc.fit_transform(X6_train)
X6_test = sc.transform(X6_test)

# Trying different models for a Better Accuracy

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier1_1 = RandomForestClassifier(n_estimators=10,random_state=0,criterion='entropy')
classifier1_1.fit(X1_train,Y1_train)

from sklearn.metrics import confusion_matrix, accuracy_score
Y1_pred = classifier1_1.predict(X1_test)
cm = confusion_matrix(Y1_test, Y1_pred)
print(cm)
accuracy_score(Y1_test, Y1_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier1_1, X = X1_train, y = Y1_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [10, 15, 20, 25], 'random_state': [0,1,5,42,75], 'criterion': ['entropy'], 
              'max_depth': [5, 8, 15, 25, 30], 'min_samples_split' :[2, 5, 10, 15, 100], 'min_samples_leaf': [1, 2, 5, 10]}]
grid_search = GridSearchCV(estimator = classifier1_1, param_grid = parameters, 
                           scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search.fit(X1_train, Y1_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# Kernel SVM Model
from sklearn.svm import SVC
classifier1_2 = SVC(kernel = 'rbf', random_state = 0)
classifier1_2.fit(X1_train, Y1_train)

Y1_pred = classifier1_2.predict(X1_test)
cm = confusion_matrix(Y1_test, Y1_pred)
print(cm)
accuracy_score(Y1_test, Y1_pred)

accuracies = cross_val_score(estimator = classifier1_2, X = X1_train, y = Y1_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier1_2,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X1_train, Y1_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# Gaussian Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
classifier1_3 = GaussianNB()
classifier1_3.fit(X1_train, Y1_train)

Y1_pred = classifier1_3.predict(X1_test)
cm = confusion_matrix(Y1_test, Y1_pred)
print(cm)
accuracy_score(Y1_test, Y1_pred)

accuracies = cross_val_score(estimator = classifier1_3, X = X1_train, y = Y1_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

parameters = [{'var_smoothing': np.logspace(0,-9, num=100)}]
grid_search = GridSearchCV(estimator = classifier1_3,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X1_train, Y1_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# Importing Test Dataset
dataset_test = pd.read_csv('C:/Users/HP/Downloads/test_data_iitm.csv')

# Data Exploration
dataset_test.shape
dataset_test.info()
dataset_test.describe()
dataset_test.dtypes

# Data manipulation
dataset_test['type'].replace(to_replace=['L','M','H'],value=[1,2,3],inplace=True)

# Adding columns (Predictions) to test dataset
# Column - 1:
classifier1 = RandomForestClassifier(n_estimators=15, random_state=0, criterion='entropy',
                                      max_depth=25, min_samples_leaf=1, min_samples_split=5)
classifier1.fit(X1_train,Y1_train)

Y1_pred = classifier1.predict(X1_test)
cm = confusion_matrix(Y1_test, Y1_pred)
print(cm)
accuracy_score(Y1_test, Y1_pred)

accuracies = cross_val_score(estimator = classifier1, X = X1_train, y = Y1_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

A1 = dataset_test.iloc[:,2:].values
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
A1 = ss.fit_transform(A1)

column_1 = classifier1.predict(A1)
dataset_test['Machine Failure'] = column_1
dataset_test.head()

# Column - 2:
classifier2 = RandomForestClassifier(n_estimators=15, random_state=0, criterion='entropy',
                                      max_depth=25, min_samples_leaf=1, min_samples_split=5)
classifier2.fit(X2_train,Y2_train)

Y2_pred = classifier2.predict(X2_test)
cm = confusion_matrix(Y2_test, Y2_pred)
print(cm)
accuracy_score(Y2_test, Y2_pred)

accuracies = cross_val_score(estimator = classifier2, X = X2_train, y = Y2_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

A2 = dataset_test.iloc[:,2:].values
A2 = ss.fit_transform(A2)

column_2 = classifier2.predict(A2)
dataset_test['Tool Wear Failure'] = column_2
dataset_test.head()

# Column - 3:
classifier3 = RandomForestClassifier(n_estimators=15, random_state=0, criterion='entropy',
                                      max_depth=25, min_samples_leaf=1, min_samples_split=5)
classifier3.fit(X3_train,Y3_train)

Y3_pred = classifier3.predict(X3_test)
cm = confusion_matrix(Y3_test, Y3_pred)
print(cm)
accuracy_score(Y3_test, Y3_pred)

accuracies = cross_val_score(estimator = classifier3, X = X3_train, y = Y3_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

A3 = dataset_test.iloc[:,2:9].values
A3 = ss.fit_transform(A3)

column_3 = classifier3.predict(A3)
dataset_test['Heat dissipation Failure'] = column_3
dataset_test.head()

# Column - 4:
classifier4 = RandomForestClassifier(n_estimators=15, random_state=0, criterion='entropy',
                                      max_depth=25, min_samples_leaf=1, min_samples_split=5)
classifier4.fit(X4_train,Y4_train)

Y4_pred = classifier4.predict(X4_test)
cm = confusion_matrix(Y4_test, Y4_pred)
print(cm)
accuracy_score(Y4_test, Y4_pred)

accuracies = cross_val_score(estimator = classifier4, X = X4_train, y = Y4_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

A4 = dataset_test.iloc[:,2:9].values
A4 = ss.fit_transform(A4)

column_4 = classifier4.predict(A4)
dataset_test['Power Failure'] = column_4
dataset_test.head()

# Column - 5:
classifier5 = RandomForestClassifier(n_estimators=15, random_state=0, criterion='entropy',
                                      max_depth=25, min_samples_leaf=1, min_samples_split=5)
classifier5.fit(X5_train,Y5_train)

Y5_pred = classifier5.predict(X5_test)
cm = confusion_matrix(Y5_test, Y5_pred)
print(cm)
accuracy_score(Y5_test, Y5_pred)

accuracies = cross_val_score(estimator = classifier5, X = X5_train, y = Y5_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

A5 = dataset_test.iloc[:,2:9].values
A5 = ss.fit_transform(A5)

column_5 = classifier5.predict(A5)
dataset_test['High Torque Failure'] = column_5
dataset_test.head()

# Column - 6:
classifier6 = RandomForestClassifier(n_estimators=15, random_state=0, criterion='entropy',
                                      max_depth=25, min_samples_leaf=1, min_samples_split=5)
classifier6.fit(X6_train,Y6_train)

Y6_pred = classifier6.predict(X6_test)
cm = confusion_matrix(Y6_test, Y6_pred)
print(cm)
accuracy_score(Y6_test, Y6_pred)

accuracies = cross_val_score(estimator = classifier6, X = X6_train, y = Y6_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

A6 = dataset_test.iloc[:,2:9].values
A6 = ss.fit_transform(A6)

column_6 = classifier6.predict(A6)
dataset_test['Random Failure'] = column_6
dataset_test.head()

# Saving the Predictions file
dataset_test.to_csv('CE18B043_Jagadeeswar_Predictions.csv')

