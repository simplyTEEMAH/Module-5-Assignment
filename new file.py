import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
cancer = datasets.load_breast_cancer()
print (cancer.DESCR) 
'''
This is the Breast Cancer Winconsin dataset with 569 instances, 30 attributes and 2 classes (WDBC malignant and WDBC benign)
Also, the description gives the summary statistics of the 30 attributes.
There are no missing attributes in the dataset
Class distribution of 212 malignant and 357 benign cases.
'''
df=cancer
df= pd.DataFrame(df.data,columns=df.feature_names) #convert the dataset into a dataframe to perform data exploration
df.head() # view the first 5 rows of the dataset
df.shape # there are 569 rows and 30 columns in the dataset
df.info() # all 30 columns have numerical variables
df.duplicated().sum() # there are no duplicates in the dataset
df.isnull().sum() # there are no missing values in the dataset
scaler=StandardScaler() # Standardise the data
scaler.fit(df) # fit the standardised data
scaled_data=scaler.transform(df) # transform the standardised data
print (scaled_data) # view the scaled dataset
pca=PCA() # apply PCA to the scaled dataset
pca.fit(scaled_data) # fit the scaled data to the PCA
df_pca=pca.transform(scaled_data) # transform the scaled data to the PCA
df_pca # prints the independent features of the principal components
explained_variance_ratio = pca.explained_variance_ratio_ # check the proportion of variance in the dataset explained by the principal components
print(f"Explained Variance Ratio: {explained_variance_ratio}")
# visualize the proportion of variance explained by each principal component.
fig, ax = plt.subplots()
# set x and y values
x = np.arange(1, len(explained_variance_ratio) + 1)
y = explained_variance_ratio
# plot
ax.plot(x, y, marker='o')
# set label and title
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('Explained Variance Ratio by Principal Component')
plt.show() # 3 principal components explain a large proportion of the variance in the dataset.
# view the different components of the dataset
components = pd.DataFrame(pca.components_, columns=df.columns)
print(components)
# view the top 5 important features for each principal components
for i in range(len(components)):
    top_features = components.iloc[i].abs().sort_values(ascending=False).head(5)
    print(f"Top 5 Features for PC{i+1}:")
    print(top_features)
    print("\n")
# Reduce the dataset into 2 PCA components
pca=PCA(n_components=2)
# fit the scaled data to the 2 principal components
pca.fit(scaled_data)
# transform the dataset
df_pca=pca.transform(scaled_data)
df_pca.shape # there are 569 rows and 2 principal components
# view the principal components of the dataset
df_pca
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: {explained_variance_ratio}")
# visualize the proportion of variance explained by each principal component.
fig, ax = plt.subplots()
# set x and y values
x = np.arange(1, len(explained_variance_ratio) + 1)
y = explained_variance_ratio
# plot
ax.plot(x, y, marker='o')
# set label and title
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('Explained Variance Ratio by Principal Component')
plt.show()
components = pd.DataFrame(pca.components_, columns=df.columns)
print(components)
# view the top 3 features for each principal components
for i in range(len(components)):
    top_features = components.iloc[i].abs().sort_values(ascending=False).head(3)  # Top 3 features by absolute loading
    print(f"Top 2 Features for PC{i+1}:")
    print(top_features)
    print("\n")
# define the X and y variables
X= df.values
y= cancer.target
X_pca=df_pca
# perform the train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=1) # data split into train test in the ratio of 70:30 and the principal components of X adopted
print("Shape of Training set : ", X_train.shape)
print("Shape of test set : ", X_test.shape)
print("Shape of Training set : ", y_train.shape)
print("Shape of test set : ", y_test.shape)
import statsmodels.api as sm
X_train_const = sm.add_constant(X_train)
logit = sm.Logit(y_train, X_train.astype(float))
lg = logit.fit(disp=False) ## Complete the code to fit logistic regression
print(lg.summary()) # result summary: x1 is negatively correlated with the target variable y while x2 is positively correlated with the target variable y. Both X1 and X2 are statistically significant given their p values.

