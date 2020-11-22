#library for map of US
import geopandas as gpd


#packages and libraries
import numpy as np
import pandas as pd 
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "La Nina"]



#import data from saved csv
df = pd.read_csv('ADD FILE PATH TO CSV')
wash = pd.DataFrame(df)


#list of years la nina occured cited in portfolip other citations
#[1903, 1906, 1909, 1916, 1924, 1928, 1938, 1950, 1954, 1964, 1970, 1973, 1975, 1988, 1995,1998, 2007, 2010]
la_nina_yrs = [1970, 1973, 1975, 1988, 1995,1998, 2007, 2010]



#for loop for firs extracting the year from the sunrise column for the binary la nina column
year = np.array(data['sunrise'])
years = []
for i in year:
    years.append(i[0:4])
wash['year'] = years
wash['year'] = wash['year'].astype(int)

#for loop to create the binary column for la nina years
la_nina_col = []
for i in wash['year']:
    if i in la_nina_yrs:
        la_nina_col.append('1')
    else:
        la_nina_col.append('0')
wash['Class'] = la_nina_col
wash['Class'] = wash['Class'].astype(int)

'''
#reclean the dates to datetime from object and set columns to strings
data['sunrise'] = pd.to_datetime(data['sunrise'], format='%Y-%m-%d %H:%M:%S')
data['sunset'] = pd.to_datetime(data['sunset'], format='%Y-%m-%d %H:%M:%S')
data['conditions'] = data['conditions'].astype('string')
data['state'] = data['state'].astype('string')
'''
data = pd.DataFrame({
    'temp' : wash['temp'].astype(float),  
    'maxt' : wash['maxt'].astype(float),  
    'wspd' : wash['wspd'].astype(float), 
    'heatindex' : wash['heatindex'].astype(float), 
    'cloudcover' : wash['cloudcover'].astype(float), 
    'mint' : wash['mint'].astype(float),  
    'precip' : wash['precip'].astype(float), 
    'weathertype' : wash['weathertype'].astype(float), 
    'moonphase' : wash['moonphase'].astype(float), 
    'snowdepth' : wash['snowdepth'].astype(float), 
    'humidity' : wash['humidity'].astype(float), 
    'wgust' : wash['wgust'].astype(float), 
    'windchill' : wash['windchill'].astype(float), 
    'lat' : wash['lat'].astype(float), 
    'long' : wash['long'].astype(float),
    'year' : wash['year'].astype(int),
    'Class' : wash['Class'].astype(int)})

#Data review steps
data.head()
data.info()
data.describe()

#Eploratory Data Analysis
data.isnull().values.any() # No null values retursn false

count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title('Transaction Class Distrobution')
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()


#Get the la_nina and the normal dataset
la_nina = data[data['Class']==1]
normal = data[data['Class']==0]
print(la_nina.shape,normal.shape)

# We need to analyze more amount of info from the transaction data
#how different are the amount of money used in different transaction classes?
la_nina.describe()
normal.describe()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Precip per Day by Class')
bins = 50
ax1.hist(la_nina.precip, bins = bins)
ax1.set_title('La Nina')
ax2.hist(normal.precip, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Precipitation in Inches')
plt.ylabel('Number of Days')
plt.xlim((0,10))
plt.yscale('log')
plt.show();




#Create a sample of the data
data1 = data.sample(frac = 0.1,random_state=1)
data1.shape

La_Nina = data1[data1['Class']==1]
Valid = data1[data1['Class']==0]
outlier_fraction = len(La_Nina)/float(len(Valid))
print(outlier_fraction)

print("La_Nina Cases : {}".format(len(La_Nina)))
print('Valid Cases : {}'.format(len(Valid)))

#Correlation
#may need to re improt seaborn as sns
#import seaborn as sns
#get correlations of each features is dataset
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(5, 5))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()

#create independent and Dependent Features
columns = data1.columns.tolist()
#filter the columns to remove data we dont want
columns = [c for c in columns if c not in ["Class"]]
#Store the variable we are predicting
target = "Class"
#define a random state
state = np.random.RandomState(42)
X = data1[columns]
Y = data1[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape(1))
print(X.shape)
print(Y.shape)


#define the outlier detection methods
classifiers = {
    
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), contamination=outlier_fraction,random_state=state, verbose=0),
    
    "Local Outlier Fractor":LocalOutlierFactor(novelty=True, n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination=outlier_fraction),

    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, max_iter=-1, random_state=state)
}

type(classifiers)


n_outliers = len(La_Nina)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #fir the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
    else:
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
#repshape the predicition values to 0 for Valid transaction, 1 for la_nina day

y_pred[y_pred == 1] = 0
y_pred[Y-pred == -1] = 1
n_errors = (y_pred != Y).sum()
#run classification metrics
print("{}: {}".format(clf_name,n_errors))
print("Accuracy Score :)
print(accuracy_score(Y,y_pred))
print("Classification Report :")
print(classification_report(Y,y_pred))

