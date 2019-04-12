# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:13:51 2019

@author: 54329
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:36:03 2019
@author: Stephy Zirit
Working Directory:
/Users/Stephy Zirit/Documents/HULT/Module B/Machine Learning
Purpose:
    Certain factors contribute to the health of a newborn baby.One such 
    health measure is birth weight. Countless studies have identified factors, 
    both preventative and hereditary, that lead to low birth weight.
    
    Your team has been hired as public health consultants to analyze and 
    model an infant’s birth weight based on such characteristics.
"""

###############################################################################
##### DATA DICTIONARY
###############################################################################
"""
|----------|---------|---------------------------------|
| variable | label   | description                     |
|----------|---------|---------------------------------|
| 1        | mage    | mother's age                    |
| 2        | meduc   | mother's educ                   |
| 3        | monpre  | month prenatal care began       |
| 4        | npvis   | total number of prenatal visits |
| 5        | fage    | father's age, years             |
| 6        | feduc   | father's educ, years            |
| 7        | omaps   | one minute apgar score          |
| 8        | fmaps   | five minute apgar score         |
| 9        | cigs    | avg cigarettes per day          |
| 10       | drink   | avg drinks per week             |
| 11       | male    | 1 if baby male                  |
| 12       | mwhte   | 1 if mother white               |
| 13       | mblck   | 1 if mother black               |
| 14       | moth    | 1 if mother is other            |
| 15       | fwhte   | 1 if father white               |
| 16       | fblck   | 1 if father black               |
| 17       | foth    | 1 if father is other            |
| 18       | bwght   | birthweight, grams              |
|----------|---------|---------------------------------|
"""

###############################################################################
##### LIBRARIES AND FILE SET UP
###############################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file = 'birthweight_feature_set.xlsx'
birth_weight = pd.read_excel(file)

birth_weight = birth_weight.drop(['omaps','fmaps'], axis=1)

###############################################################################
##### DESCRIPTIVES STATISTICS
###############################################################################
# Column names
birth_weight.columns

# Displaying the first rows of the DataFrame
print(birth_weight.head())

# Dimensions of the DataFrame
birth_weight.shape

# Information about each variable
birth_weight.info()

# Descriptive statistics
birth_weight.describe().round(2)

###############################################################################
##### MISSING VALUES
###############################################################################
print(birth_weight.isnull().sum()) 

# Flagging missing values
for col in birth_weight:
    if birth_weight[col].isnull().astype(int).sum() > 0:
        birth_weight['m_'+col] = birth_weight[col].isnull().astype(int)

# Filling NAs in 'npvis' , 'meduc' and 'feduc' with their MEDIANs
birth_weight.npvis = birth_weight.npvis.fillna(birth_weight.npvis.median())
birth_weight.meduc = birth_weight.meduc.fillna(birth_weight.meduc.median())
birth_weight.feduc = birth_weight.feduc.fillna(birth_weight.feduc.median())

# Rechecking NAs:
print(birth_weight.isnull().sum()) 

###############################################################################
##### EXPLORATORY ANALYSIS
###############################################################################
## Histograms to check distributions:
for col in birth_weight.columns:
    x = birth_weight[col]
    plt.title("Variable: "+col)
    plt.hist(x)
    plt.show()

# Boxplots for numerical variables: 
for col in birth_weight.columns[:8]:
    x = birth_weight[col]
    plt.title("Variable: "+col)
    plt.boxplot(x,vert=False)
    plt.show()

## Correlation between variables:
# adding jitter to better visualize data
def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

for col in birth_weight.columns:
    x = birth_weight[col]
    y = birth_weight['bwght']
    plt.scatter(rand_jitter(x), y)
    plt.xlabel(col)
    plt.ylabel("Birth weight")
    plt.axhline(2500,color='blue')
    plt.axhline(4000,color='red')
    plt.show()

# Correlation matrix
birth_weight.corr().round(2)

# Correlations with birth weight
birth_weight.corr()['bwght'].sort_values()

# Classification: low, normal, high weight
df = pd.DataFrame.copy(birth_weight) #create df just for weight classification

df['wclass'] = 'norm_weight'
df.loc[df.bwght < 2500,'wclass'] = 'lo_weight'
df.loc[df.bwght > 4000,'wclass'] = 'hi_weight'

weights = pd.get_dummies(df['wclass'],drop_first=False)
df = pd.concat([df,weights],axis=1)

# Paiwise relationship:
for col1 in range(0,len(df.columns[:16])):
    x = df.columns[col1]
    for col2 in range(0,len(df.columns[:16])):
        y = df.columns[col2]
        if x != 'wclass':
            if y != 'wclass':
                sns.lmplot(x,y,data=df,hue='wclass',fit_reg=False)
                #plt.savefig(str(col1)+'_'+str(col2)+'.png')
                plt.show()

################
## PIVOT TABLES

pivot_vals = ['mage',
              'meduc',
              'monpre',
              'npvis',
              'fage',
              'feduc',
              'cigs',
              'drink',
              'male',
              'mwhte',
              'mblck',
              'moth',
              'fwhte',
              'fblck',
              'foth',
              'bwght']

table_mean = pd.pivot_table(
                        df, values=pivot_vals, index='wclass',
                        aggfunc=np.mean).round(2).iloc[[1,2,0],:]
print(table_mean)

table_median = pd.pivot_table(
                        df,values=pivot_vals, index='wclass',
                        aggfunc=np.median).round(2).iloc[[1,2,0],:]
print(table_median)

###############################################################################
##### BUILDING NEW VARIABLES
###############################################################################
# Creating binary variable 'drinker'
birth_weight['drinker'] = (birth_weight.drink > 0).astype('int')

# Creating binary variable 'smoker'
birth_weight['smoker'] = (birth_weight.cigs > 0).astype('int')

# Combination of drinker and smoker
birth_weight['trasher'] = birth_weight.drinker+birth_weight.smoker
birth_weight.loc[birth_weight.trasher == 2,'trasher'] = 4

# Binary outliers:
# Creating binary variable 'out_drink' # original=12
birth_weight['out_drink'] = (birth_weight.drink > 8).astype('int')

# Creating binary variable 'out_cigs' # original - no outliers?
birth_weight['out_cigs'] = (birth_weight.cigs > 12).astype('int')

# Creating binary variable 'lo_out_npvis' # original=7
birth_weight['lo_out_npvis'] = (birth_weight.npvis <=10).astype('int')

# Creating binary variable 'hi_out_npvis' # original=15
birth_weight['hi_out_npvis'] = (birth_weight.npvis > 15).astype('int')

# Creating binary variable 'out_mage' # original=60 (2nd = 54)
birth_weight['mover_54'] = (birth_weight.mage > 54).astype('int')

# Creating binary variable 'out_fage' # original=55 (2nd = 46)
birth_weight['fover_46'] = (birth_weight.fage > 46).astype('int')

# Creating binary variable 'mcollege' - mothers who went to college
birth_weight['mcollege'] = (birth_weight.meduc >= 14).astype('int')

# Creating binary variable 'fcollege' - fathers who went to college
birth_weight['fcollege'] = (birth_weight.feduc >= 14).astype('int')

# Getting the log value of father's education because of skewness 
birth_weight['log_feduc']= np.log(birth_weight['feduc'])

# Combine ages of mather and father
""" 
mage (20-29) & fage (20-64) - standard 1
mage (30-34) & fage (20-39) - standard 1
mage (30-34) & fage (40-64) - high risk 2
mage (35-44) & fage (35-39) - high risk 2
mage (35-44) & fage (40-64) - highest risk 3
else - highest risk 3
"""
counter = 0
birth_weight['cage'] = 0

for value in birth_weight['mage']:
    if value < 30:
        if birth_weight.loc[counter, 'fage'] < 65:
            birth_weight.loc[counter,'cage'] = 1
        elif birth_weight.loc[counter,'fage'] >= 65:
            birth_weight.loc[counter,'cage'] = 2      
    elif value < 35:
        if birth_weight.loc[counter, 'fage'] < 40:
            birth_weight.loc[counter,'cage'] = 1
        elif birth_weight.loc[counter,'fage'] < 65:
            birth_weight.loc[counter,'cage'] = 2 
        elif birth_weight.loc[counter,'fage'] >= 65:
            birth_weight.loc[counter,'cage'] = 3
    elif value < 45:
        if birth_weight.loc[counter,'fage'] < 40:
            birth_weight.loc[counter,'cage'] = 2 
        elif birth_weight.loc[counter,'fage'] >= 40:
            birth_weight.loc[counter,'cage'] = 3
    else:
        birth_weight.loc[counter,'cage'] = 3
    counter += 1

""" Month prenatal care started - avg number of visits - regular range
month 1 - 14 - (8,15)
month 2 - 13 - (7,14)
month 3 - 12 - (6,13)
month 4 - 11 - (5,12)
month 5 - 10 - (4,11)
month 6 - 8 - (3,9)
month 7 - 6 - (2,7)
month 8 - 4 - (1,5)
"""   
counter = 0
birth_weight['regular'] = 0

for value in birth_weight['npvis']:
    if birth_weight.loc[counter,'monpre'] == 1:
        if (value >= 8 and value <= 15):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 2:
        if (value >= 7 and value <= 14):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 3:
        if (value >= 6 and value <= 13):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 4:
        if (value >= 5 and value <= 12):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 5:
        if (value >= 4 and value <= 11):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 6:
        if (value >= 3 and value <= 9):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 7:
        if (value >= 2 and value <= 7):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 8:
        if (value >= 1 and value <= 5):
            birth_weight.loc[counter,'regular'] = 1 
    counter += 1

# Checking correlation with new variables
birth_weight.corr()['bwght'].sort_values().round(2)

# Export file to excel
birth_weight.to_excel('birthweight_filled.xlsx')

###############################################################################
##### IMPORT ADITIONAL LIBRARIES FOR MODELING
###############################################################################

import statsmodels.formula.api as smf 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 

###############################################################################
##### LINEAR REGRESSION USING STATS MODELS
###############################################################################
df = pd.DataFrame.copy(birth_weight)

# Building a Regression Base
lm_babyweight = smf.ols(formula = """bwght ~ df['mage'] +
                        df['meduc'] +
                        df['monpre'] +
                        df['npvis'] +
                        df['fage'] +
                        df['feduc'] +
                        df['cigs'] +
                        df['drink'] +
                        df['male'] +
                        df['mwhte'] +
                        df['mblck'] +
                        df['moth'] +
                        df['fwhte'] +
                        df['fblck'] +
                        df['foth'] +
                        df['drinker'] +
                        df['smoker'] + 
                        df['trasher'] +
                        df['out_drink'] +
                        df['out_cigs'] +
                        df['lo_out_npvis'] +
                        df['hi_out_npvis'] +
                        df['mover_54'] +
                        df['fover_46'] +
                        df['mcollege'] +
                        df['fcollege'] +
                        df['log_feduc'] +
                        df['cage'] +
                        df['regular'] """,
                        data = df)

# Fitting Results
results = lm_babyweight.fit()

# Printing Summary Statistics
print(results.summary())

###############################################################################
##### TESTING KNN
###############################################################################

def test_knn(variables,knn):
    bb_data = df.loc[:,variables]
    
    bb_target = df.loc[:,'bwght']

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
                                                        bb_data,bb_target, 
                                                        test_size = 0.1, 
                                                        random_state=508)
    for k in range(1,knn):
        print('Number of neighbors : '+str(k))
        bb = KNeighborsRegressor(algorithm = 'auto',
                                  n_neighbors = k)
        bb.fit(Xf_train,yf_train)

        # Compute and print R^2 and RMSE
        yf_pred_reg2 = bb.predict(Xf_test)
        print('\tR-Squared (train set): ',bb.score(Xf_train,yf_train).round(3))
        print('\tR-Squared  (test set): ',bb.score(Xf_test,yf_test).round(3))
        rmse = np.sqrt(mean_squared_error(yf_test , yf_pred_reg2))
        print("\tRoot Mean Squared Error: {}\n".format(rmse))

# KNN Model with Rˆ2 = 0.614 with k=7
test_knn(['mage','fage','cigs','drink','npvis','lo_out_npvis','out_drink'], 15)

###############################################################################
##### TEST MODELS FUNCTION
###############################################################################


# Define function to test different models
def test_regression(variables):
    values = str()
    for var in variables: 
        if var == variables[-1]:
            values = values + "birth_weight['"+var+"']"
        else:
            values = values + "birth_weight['"+var+"'] +"

    baby_ols = smf.ols(formula = "bwght ~ "+values, data=birth_weight)
    baby_fit = baby_ols.fit()

    print(baby_fit.summary())
    print("\n####################################\n")
    print("## Testing variables in sklearn:\n")
    
    bb_data = birth_weight.loc[:,variables]
    bb_target = birth_weight.loc[:,'bwght']
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
                                                        bb_data,bb_target, 
                                                        test_size=0.1, 
                                                        random_state=508)
    bb_lr = LinearRegression()
    bb_fit = bb_lr.fit(Xf_train, yf_train)

    # Compute and print R^2 and RMSE
    yf_pred_reg = bb_fit.predict(Xf_test)
    print('\tR-Squared: ', baby_fit.rsquared.round(3))
    print('\tTraining score: ', bb_fit.score(Xf_train,yf_train).round(3))
    print('\tTesting Score: ', bb_fit.score(Xf_test,yf_test).round(3))
    rmse = np.sqrt(mean_squared_error(yf_test , yf_pred_reg))
    print("\tRoot Mean Squared Error: {}".format(rmse))


# Using testing function:
# All variables
test_regression(['mage', 'meduc', 'monpre', 'npvis', 'fage', 'feduc', 'cigs', 
                 'drink','male', 'mwhte', 'mblck', 'moth', 'fwhte', 'fblck', 
                 'foth','drinker', 'smoker', 'trasher', 'out_drink', 
                 'out_cigs', 'lo_out_npvis', 'hi_out_npvis', 'mover_54',
                 'fover_46', 'mcollege', 'fcollege', 'log_feduc', 'cage', 
                 'regular'])

# Testing only wiht significant variables
test_regression(['mage', 'cigs','drink', 'mwhte', 'mblck', 'moth', 'fwhte', 
                 'fblck', 'foth'])
    
# Significant and more correlated variables + outliers
test_regression(['mage','cigs','drink','lo_out_npvis',
                 'hi_out_npvis','out_drink','drinker'])
# Final model
test_regression(['cigs','drink','mover_54',
                 'fage','log_feduc'])


  
###############################################################################
##### FINAL MODEL - BEST OVERALL
###############################################################################
""" 
Analyst:
    After trying different models we chose the one that perfomed better overall
"""

# Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 

file = 'birthweight_filled.xlsx'
birth_weight = pd.read_excel(file)

# Define function to test different models
def test_regression(variables):
    values = str()
    for var in variables: 
        if var == variables[-1]:
            values = values + "birth_weight['"+var+"']"
        else:
            values = values + "birth_weight['"+var+"'] +"

    baby_ols = smf.ols(formula = "bwght ~ "+values, data=birth_weight)
    baby_fit = baby_ols.fit()

    print(baby_fit.summary())
    print("\n####################################\n")
    print("## Testing variables in sklearn:\n")
    
    bb_data = birth_weight.loc[:,variables]
    bb_target = birth_weight.loc[:,'bwght']
    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
                                                        bb_data,bb_target, 
                                                        test_size=0.1, 
                                                        random_state=508)
    bb_lr = LinearRegression()
    bb_fit = bb_lr.fit(Xf_train, yf_train)

    # Compute and print R^2 and RMSE
    yf_pred_reg = bb_fit.predict(Xf_test)
    print('\tR-Squared: ', baby_fit.rsquared.round(3))
    print('\tTraining score: ', bb_fit.score(Xf_train,yf_train).round(3))
    print('\tTesting Score: ', bb_fit.score(Xf_test,yf_test).round(3))
    rmse = np.sqrt(mean_squared_error(yf_test , yf_pred_reg))
    print("\tRoot Mean Squared Error: {}".format(rmse))
    

test_regression(['cigs','drink','mover_54','fage','log_feduc'])

# Extracting predictions
best_variables = ['cigs','drink','mover_54','fage','log_feduc']

baby_data = birth_weight.loc[:, best_variables]
    
baby_target = birth_weight.loc[:,'bwght']

X0_train, X0_test, y0_train, y0_test = train_test_split(
                                                        baby_data,baby_target, 
                                                        test_size = 0.1, 
                                                        random_state=508)
best_model = LinearRegression()
best_model.fit(X0_train, y0_train)

y0_pred = best_model.predict(X0_test)

final_predictions = pd.DataFrame(y0_pred)

final_predictions.to_excel('predictions.xlsx')  