#!/usr/bin/env python
# coding: utf-8

# In[132]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[82]:


df = pd.read_csv('dataset.csv')
df


# In[83]:


df.shape


# In[84]:


df.info()


# In[85]:


df.hist(figsize=(12,12), layout=(5,3));


# In[86]:


sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')


# In[87]:


df.head()


# In[88]:


df.tail()


# In[89]:


df.isnull().sum()


# In[90]:


# statistical measures about the data
df.describe()


# In[91]:


import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[92]:


X = df.drop(columns='target', axis=1)
Y = df['target']


# In[93]:


print(X)


# In[94]:


print(Y)


# In[95]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=40)


# In[96]:



# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(Y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
sns.heatmap(cm, annot=True,cmap='BuPu')


# In[97]:


# Import necessary libraries
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(Y_test, y_pred)

# Extract TP, TN, FP, FN
TP = cm[0][0]
TN = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

# Calculate accuracy
accuracy = (TP + TN) / (TP + TN + FN + FP)

# Calculate precision
precision = TP / (TP + FP)

# Calculate recall
recall = TP / (TP + FN)

# Calculate F1 score
f1_score = 2 * (precision * recall) / (precision + recall)

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# Create a histogram
metrics = [accuracy, precision, recall, f1_score]
labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

plt.figure(figsize=(8, 6))
plt.bar(labels, metrics, color=['green', 'red', 'yellow', 'blue'])
plt.ylabel("Value")
plt.title("Metrics")
plt.show()


# In[98]:


from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(Y_test, y_pred)

# Calculate the AUC (Area Under the Curve)
roc_auc = roc_auc_score(Y_test, y_pred)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# Logistic Regression
# 

# In[105]:


log = LogisticRegression()
log.fit(X_train,Y_train)
y_pred1 = log.predict(X_test)
accuracy_score(Y_test,y_pred1)


# Decision Tree

# In[108]:


dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)
y_pred2= dt.predict(X_test)
accuracy_score(Y_test,y_pred2)


# Random Forest

# In[110]:


rf = RandomForestClassifier()
rf.fit(X_train,Y_train)
y_pred3= rf.predict(X_test)
accuracy_score(Y_test,y_pred3)


# Support Vector Machine

# In[133]:



svm = SVC()
svm.fit(X_train,Y_train)
y_pred4 = svm.predict(X_test)
accuracy_score(Y_test,y_pred4)


# K nearest neighnor

# In[115]:


# Create and train a KNN classifier
k = 3  # Choose the number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, Y_train)

# Make predictions on the test set
y_pred5 = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, y_pred5)
print("Accuracy using KNN:", accuracy)


# Naive Bayes

# In[128]:


from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(X_train, Y_train)
y_pred6  = NB.predict(X_test)
accuracy_score(Y_test,y_pred6)


# In[138]:


final_data = pd.DataFrame({'Models':['LR','DT','RF','SVM','KNN','NB'],
                          'ACC':[accuracy_score(Y_test,y_pred1)*100,
                                accuracy_score(Y_test,y_pred2)*100,
                                accuracy_score(Y_test,y_pred3)*100,
                                accuracy_score(Y_test,y_pred4)*100,
                                accuracy_score(Y_test,y_pred5)*100,
                                accuracy_score(Y_test,y_pred6)*100]})
final_data


# In[139]:


sns.barplot(final_data['Models'],final_data['ACC'])


# In[142]:



# Assuming the dataset has 12 columns and you want to reduce it to 5 dimensions
n_components = 5

# Create a PCA instance and fit it to the data
pca = PCA(n_components=n_components)
pca.fit(data)

# Transform the data to the reduced dimensionality
data_reduced = pca.transform(df)

# Create a new DataFrame with the reduced dimensions
reduced_df = pd.DataFrame(data_reduced, columns=[f'PC{i+1}' for i in range(n_components)])

# Print the first few rows of the reduced data
print(reduced_df.head())

# Plot the data
plt.figure(figsize=(8, 6))
for i in range(n_components):
    for j in range(i + 1, n_components):
        plt.scatter(reduced_df[f'PC{i + 1}'], reduced_df[f'PC{j + 1}'], label=f'PC{i + 1} vs PC{j + 1}')

plt.xlabel(f'Principal Component 1')
plt.ylabel(f'Principal Component 2')
plt.title(f'Scatter Plot of Principal Components')
plt.legend()
plt.show()


# In[ ]:





# In[160]:


'''
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
'''
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train the Logistic Regression classifier
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = lr.predict(X_test)

# Calculate accuracy
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

# Initialize and train the Decision Tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt.predict(X_test)

# Calculate accuracy
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

# Initialize and train the Random Forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf.predict(X_test)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Initialize and train the SVM classifier
svm = SVC()
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = svm.predict(X_test)

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred_knn = knn.predict(X_test)

# Calculate accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)

# Initialize and train the Naïve Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)

# Make predictions on the test set
y_pred_nb = nb.predict(X_test)

# Calculate accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naïve Bayes Accuracy:", accuracy_nb)

# You now have y_pred_lr, y_pred_dt, y_pred_rf, y_pred_svm, y_pred_knn, and y_pred_nb


# In[161]:


import matplotlib.pyplot as plt

# Collect the accuracy values for each classifier
accuracies = [accuracy_lr, accuracy_dt, accuracy_rf, accuracy_svm, accuracy_knn, accuracy_nb]
classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Naïve Bayes']

# Create a bar chart to show the accuracy of each classifier
plt.figure(figsize=(10, 6))
plt.bar(classifiers, accuracies, color=['green', 'red', 'blue', 'yellow', 'purple', 'orange'])
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Classifiers')
plt.ylim(0, 1)  # Set the y-axis limit to the range [0, 1]
plt.xticks(rotation=45)  # Rotate the x-axis labels for better visibility
plt.show()


# In[196]:




# Create a PCA instance and fit it to the data with n_components=3
n_components = 4
pca = PCA(n_components=n_components)
pca.fit(df)

#print(df)

# Transform the data to the reduced dimensionality
data_reduced = pca.transform(df)

# Create a new DataFrame with the reduced dimensions and name it reduce_df1
reduce_df1 = pd.DataFrame(data_reduced, columns=[f'PC{i+1}' for i in range(n_components)])

# Print the first few rows of the reduced data
print(reduce_df1.head())

# Plot the data
plt.figure(figsize=(8, 6))
for i in range(n_components):
    for j in range(i + 1, n_components):
        plt.scatter(reduce_df1[f'PC{i + 1}'], reduce_df1[f'PC{j + 1}'], label=f'PC{i + 1} vs PC{j + 1}')

plt.xlabel(f'Principal Component 1')
plt.ylabel(f'Principal Component 2')
plt.title(f'Scatter Plot of Principal Components')
plt.legend()
plt.show()


# In[197]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train the Logistic Regression classifier
lr1 = LogisticRegression()
lr1.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr1 = lr1.predict(X_test)

# Calculate accuracy
accuracy_lr1 = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr1)

# Initialize and train the Decision Tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt.predict(X_test)

# Calculate accuracy
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

# Initialize and train the Random Forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf.predict(X_test)

# Calculate accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Initialize and train the SVM classifier
svm = SVC()
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = svm.predict(X_test)

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred_knn = knn.predict(X_test)

# Calculate accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)

# Initialize and train the Naïve Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)

# Make predictions on the test set
y_pred_nb = nb.predict(X_test)

# Calculate accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naïve Bayes Accuracy:", accuracy_nb)

# You now have y_pred_lr, y_pred_dt, y_pred_rf, y_pred_svm, y_pred_knn, and y_pred_nb


# In[198]:


import matplotlib.pyplot as plt

# Collect the accuracy values for each classifier
accuracies = [accuracy_lr, accuracy_dt, accuracy_rf, accuracy_svm, accuracy_knn, accuracy_nb]
classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Naïve Bayes']

# Create a bar chart to show the accuracy of each classifier
plt.figure(figsize=(10, 6))
plt.bar(classifiers, accuracies, color=['green', 'red', 'blue', 'yellow', 'purple', 'orange'])
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Classifiers')
plt.ylim(0, 1)  # Set the y-axis limit to the range [0, 1]
plt.xticks(rotation=45)  # Rotate the x-axis labels for better visibility
plt.show()


# In[ ]:





# In[218]:


'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
'''
# Load your data from dataset.csv into a pandas DataFrame
#data = pd.read_csv("dataset.csv")

# Assuming you want to reduce the data to 3 dimensions
n_components = 1
lda = LDA(n_components=n_components)
lda.fit(df.drop(columns=["target"]), data["target"])

# Transform the data to the reduced dimensionality
data_reduced2 = lda.transform(df.drop(columns=["target"]))

# Create a new DataFrame with the reduced dimensions
reduced_df3 = pd.DataFrame(data_reduced2, columns=[f'LD{i+1}' for i in range(n_components)])

# Split the data into training and testing sets
X = reduced_df3
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Initialize and train the Logistic Regression classifier
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

# Initialize and train the Decision Tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

# Initialize and train the Random Forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Initialize and train the SVM classifier
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)

# Initialize and train the Naïve Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naïve Bayes Accuracy:", accuracy_nb)


# In[231]:


'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
'''
# Load your data from dataset.csv into a pandas DataFrame (assuming it is stored in 'df.csv')
df1 = pd.read_csv("dataset.csv")

# Assuming you want to reduce the data to 1 dimension using LDA
n_components = 1
lda = LDA(n_components=n_components)
lda.fit(df1.drop(columns=["target"]), df1["target"])

# Transform the data to the reduced dimensionality
data_reduced2 = lda.transform(df1.drop(columns=["target"]))

# Create a new DataFrame with the reduced dimension
reduced_df2 = pd.DataFrame(data_reduced2, columns=['LD1'])

# Plot the LDA values
plt.figure(figsize=(8, 6))
plt.scatter(reduced_df2, df1["target"], label="LDA Values")
plt.xlabel("LD1 (Linear Discriminant 1)")
plt.ylabel("Target")
plt.title("Scatter Plot of LDA Values")
plt.legend()
plt.show()

# Split the data into training and testing sets
X = reduced_df2
y = df1["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Initialize and train the Logistic Regression classifier
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

# Initialize and train the Decision Tree classifier
dt = DecisionTreeClassifier()
dt .fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

# Initialize and train the Random Forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Initialize and train the SVM classifier
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)

# Initialize and train the Naïve Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naïve Bayes Accuracy:", accuracy_nb)

# Create a histogram of accuracy values
accuracies = [accuracy_lr, accuracy_dt, accuracy_rf, accuracy_svm, accuracy_knn, accuracy_nb]
classifiers = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "Naïve Bayes"]
plt.figure(figsize=(8, 6))
plt.bar(classifiers, accuracies, color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title("Accuracy for Different Classifiers")
plt.show()


# In[ ]:





# In[235]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load your data from dataset.csv into a pandas DataFrame
df = pd.read_csv("dataset.csv")

# Assuming you want to reduce the data to 5 dimensions using ICA
n_components = 7
ica = FastICA(n_components=n_components)
ica_data = ica.fit_transform(df.drop(columns=["target"]))

# Create a new DataFrame with the reduced dimensions
reduced_df = pd.DataFrame(ica_data, columns=[f'ICA{i+1}' for i in range(n_components)])

# Split the data into training and testing sets
X = reduced_df
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Initialize and train the Logistic Regression classifier
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

# Initialize and train the Decision Tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

# Initialize and train the Random Forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Initialize and train the SVM classifier
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)

# Initialize and train the Naïve Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naïve Bayes Accuracy:", accuracy_nb)

# Create a histogram of accuracy values
accuracies = [accuracy_lr, accuracy_dt, accuracy_rf, accuracy_svm, accuracy_knn, accuracy_nb]
classifiers = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "Naïve Bayes"]
plt.figure(figsize=(8, 6))
plt.bar(classifiers, accuracies, color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title("Accuracy for Different Classifiers")
plt.show()


# In[236]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load your data from dataset.csv into a pandas DataFrame
df = pd.read_csv("dataset.csv")

# Assuming you want to reduce the data to 5 dimensions using ICA
n_components = 7
ica = FastICA(n_components=n_components)
ica_data = ica.fit_transform(df.drop(columns=["target"]))

# Create a new DataFrame with the reduced dimensions
reduced_df = pd.DataFrame(ica_data, columns=[f'ICA{i+1}' for i in range(n_components)])

# Plot the ICA values
plt.figure(figsize=(8, 6))
for i in range(n_components):
    plt.scatter(reduced_df[f'ICA{i+1}'], df["target"], label=f'ICA{i+1} Values')

plt.xlabel("ICA Components")
plt.ylabel("Target")
plt.title("Scatter Plot of ICA Values")
plt.legend()
plt.show()

# Split the data into training and testing sets
X = reduced_df
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Initialize and train the Logistic Regression classifier
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

# Initialize and train the Decision Tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

# Initialize and train the Random Forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)

# Initialize and train the SVM classifier
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", accuracy_svm)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)

# Initialize and train the Naïve Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naïve Bayes Accuracy:", accuracy_nb)

# Create a histogram of accuracy values
accuracies = [accuracy_lr, accuracy_dt, accuracy_rf, accuracy_svm, accuracy_knn, accuracy_nb]
classifiers = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "Naïve Bayes"]
plt.figure(figsize=(8, 6))
plt.bar(classifiers, accuracies, color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title("Accuracy for Different Classifiers")
plt.show()


# In[254]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pyswarm import pso  # You will need to install the 'pyswarm' library

# Load your dataset (replace 'dataset.csv' with your actual dataset path)
data = pd.read_csv('dataset.csv')

# Step 1: Define a function for PSO optimization
def optimize_data(params):
    # Apply PSO optimization to the dataset
    # 'params' are the optimization parameters you want to tune
    # The optimized data should be stored in 'optimized_data'
    return mean_squared_error(original_data, optimized_data)  # You need to define the error function

# Replace 'original_data' with your dataset and 'optimized_data' with the result of PSO optimization
original_data = data.copy()
optimized_data = data.copy()  # Placeholder, replace with actual optimized data

# Step 2: Apply PCA for dimensionality reduction
n_components = 5

pca = PCA(n_components=n_components)
pca.fit(optimized_data)  # Assuming you've optimized the data

data_reduced = pca.transform(optimized_data)
reduced_df = pd.DataFrame(data_reduced, columns=[f'PC{i+1}' for i in range(n_components)])

# Step 3: Split the data into train and test sets
X = reduced_df  # Features
y = data['target']  # Replace 'target_column' with the actual target column name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Initialize and train the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naïve Bayes": GaussianNB()
}

accuracies = {}

for classifier_name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[classifier_name] = accuracy
    print(f"{classifier_name} Accuracy: {accuracy}")

# Create a histogram of accuracy values
plt.figure(figsize=(12, 6))  # Create a larger figure for multiple subplots

# Subplot 1: Accuracy histogram
plt.subplot(1, 2, 1)
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title("Accuracy for Different Classifiers")

# Subplot 2: PCA graph
plt.subplot(1, 2, 2)
for i in range(n_components):
    for j in range(i + 1, n_components):
        plt.scatter(reduced_df[f'PC{i + 1}'], reduced_df[f'PC{j + 1}'], label=f'PC{i + 1} vs PC{j + 1}')

plt.xlabel(f'Principal Component 1')
plt.ylabel(f'Principal Component 2')
plt.title(f'Scatter Plot of Principal Components')
plt.legend()

plt.tight_layout()  # Adjust subplot spacing
plt.show()


# In[ ]:





# In[249]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load your dataset (replace 'dataset.csv' with your actual dataset path)
data = pd.read_csv('dataset.csv')

# Step 1: Apply CSO Optimization (or PSO) or any other optimization technique to the dataset
# Replace 'optimized_data' with the result of optimization

# Step 2: Apply LDA for dimensionality reduction
n_components = 1

# Assuming 'target_column' is your target variable
X = optimized_data.drop(columns=['target'])
y = optimized_data['target']

lda = LinearDiscriminantAnalysis(n_components=n_components)
X_lda = lda.fit_transform(X, y)

reduced_df = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(n_components)])

# Step 3: Split the data into train and test sets
X = reduced_df  # Features
y = data['target']  # Replace 'target_column' with the actual target column name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Initialize and train the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naïve Bayes": GaussianNB()
}

accuracies = {}

for classifier_name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[classifier_name] = accuracy
    print(f"{classifier_name} Accuracy: {accuracy}")

# Create a histogram of accuracy values
plt.figure(figsize=(8, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title("Accuracy for Different Classifiers")
plt.show()


# In[ ]:





# In[257]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load your dataset (replace 'dataset.csv' with your actual dataset path)
data = pd.read_csv('dataset.csv')

# Step 1: Apply CSO Optimization (or PSO) or any other optimization technique to the dataset
# Replace 'optimized_data' with the result of optimization

# Step 2: Apply LDA for dimensionality reduction
n_components = 1

# Assuming 'target_column' is your target variable
X = optimized_data.drop(columns=['target'])
y = optimized_data['target']

lda = LinearDiscriminantAnalysis(n_components=n_components)
X_lda = lda.fit_transform(X, y)

reduced_df = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(n_components)])

# Step 3: Split the data into train and test sets
X = reduced_df  # Features
y = data['target']  # Replace 'target_column' with the actual target column name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Initialize and train the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naïve Bayes": GaussianNB()
}

accuracies = {}

for classifier_name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[classifier_name] = accuracy
    print(f"{classifier_name} Accuracy: {accuracy}")

# Create a subplot to display LDA graph and accuracy histogram
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot the LDA graph
ax_lda = axes[0]
ax_lda.scatter(X_lda, y, c=y, cmap='viridis')
ax_lda.set_xlabel('LD1')
ax_lda.set_title('LDA')

# Plot the accuracy histogram
ax_hist = axes[1]
ax_hist.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
ax_hist.set_xlabel("Classifiers")
ax_hist.set_ylabel("Accuracy")
ax_hist.set_title("Accuracy for Different Classifiers")

plt.tight_layout()
plt.show()


# In[268]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pyswarm import pso  # You will need to install the 'pyswarm' library for PSO optimization

# Load your dataset (replace 'dataset.csv' with your actual dataset path)
data = pd.read_csv('dataset.csv')

# Step 1: Apply PSO Optimization to the dataset
# Define your PSO optimization function and optimize your dataset
def pso_optimization(params):
    # Implement your PSO optimization here
    # You should return the optimized data as a numpy array
    optimized_data = np.array(params)
    return optimized_data

# Define the number of optimization parameters
num_parameters = 10  # Replace with the actual number of parameters

# Initialize bounds (lb and ub) for PSO; use appropriate bounds for your optimization
lb = [0.0] * num_parameters
ub = [1.0] * num_parameters

# Call PSO optimization to obtain the optimized_data
xopt, fopt = pso(pso_optimization, lb, ub)

# Step 2: Apply LDA for dimensionality reduction
n_components = 1

# Assuming 'target_column' is your target variable
X = xopt  # Use the optimized data
y = data['target']

lda = LinearDiscriminantAnalysis(n_components=n_components)
X_lda = lda.fit_transform(X, y)

# Step 3: Split the data into train and test sets
X = X_lda  # Features after LDA
y = data['target']  # Replace 'target_column' with the actual target column name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Initialize and train the classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naïve Bayes": GaussianNB()
}

accuracies = {}

for classifier_name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[classifier_name] = accuracy
    print(f"{classifier_name} Accuracy: {accuracy}")

# Create a subplot to display LDA graph and accuracy histogram
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot the LDA graph
ax_lda = axes[0]
ax_lda.scatter(X_lda, y, c=y, cmap='viridis')
ax_lda.set_xlabel('LD1')
ax_lda.set_title('LDA')

# Plot the accuracy histogram
ax_hist = axes[1]
ax_hist.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
ax_hist.set_xlabel("Classifiers")
ax_hist.set_ylabel("Accuracy")
ax_hist.set_title("Accuracy for Different Classifiers")

plt.tight_layout()
plt.show()


# In[280]:


X=data.drop('target',axis=1)
y=data['target']
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X,y)
import pandas as pd
new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
     'slope':2,
    'ca':2,
    'thal':3,    
},index=[0])


# In[281]:


new_data


# In[282]:


p = rf.predict(new_data)
if p[0]==0:
    print("Not Affected in CardionVascular Disease")
else:
    print("Affected in CardionVascular Disease")


# In[284]:


new_data = pd.DataFrame({
    'age':63,
    'sex':1,
    'cp':3,
    'trestbps':145,
    'chol':233,
    'fbs':1,
    'restecg':0,
    'thalach':150,
    'exang':0,
    'oldpeak':2.3,
     'slope':0,
    'ca':0,
    'thal':1,    
},index=[0])


# In[285]:


new_data


# In[286]:


p = rf.predict(new_data)
if p[0]==0:
    print("Not Affected in CardionVascular Disease")
else:
    print("Affected in CardionVascular Disease")


# In[287]:


import joblib


# In[288]:


joblib.dump(rf,'model_joblib_heart')


# In[289]:


model = joblib.load('model_joblib_heart')


# In[290]:


model.predict(new_data)


# In[291]:


from tkinter import *
import joblib


# In[292]:


from tkinter import *
import joblib
import numpy as np
from sklearn import *
def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=int(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=int(e8.get())
    p9=int(e9.get())
    p10=float(e10.get())
    p11=int(e11.get())
    p12=int(e12.get())
    p13=int(e13.get())
    model = joblib.load('model_joblib_heart')
    result=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p8,p10,p11,p12,p13]])
    
    if result == 0:
        Label(master, text="No Heart Disease").grid(row=31)
    else:
        Label(master, text="Possibility of Heart Disease").grid(row=31)
    
    
master = Tk()
master.title("Heart Disease Prediction System")


label = Label(master, text = "Heart Disease Prediction System"
                          , bg = "black", fg = "white"). \
                               grid(row=0,columnspan=2)


Label(master, text="Enter Your Age").grid(row=1)
Label(master, text="Male Or Female [1/0]").grid(row=2)
Label(master, text="Enter Value of CP").grid(row=3)
Label(master, text="Enter Value of trestbps").grid(row=4)
Label(master, text="Enter Value of chol").grid(row=5)
Label(master, text="Enter Value of fbs").grid(row=6)
Label(master, text="Enter Value of restecg").grid(row=7)
Label(master, text="Enter Value of thalach").grid(row=8)
Label(master, text="Enter Value of exang").grid(row=9)
Label(master, text="Enter Value of oldpeak").grid(row=10)
Label(master, text="Enter Value of slope").grid(row=11)
Label(master, text="Enter Value of ca").grid(row=12)
Label(master, text="Enter Value of thal").grid(row=13)



e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)



Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()


# In[ ]:




