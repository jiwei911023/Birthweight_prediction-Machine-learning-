#!/usr/bin/env python
# coding: utf-8

# #########################################################################
# # Created by: Valentino Gaffuri Bedetta - Gabriela Teixeira Sondermann
#               Michelle Kirsten Dionisio - Ankur Malik - JiWei Li 
# #########################################################################
# # On date 09-03-2019
# #########################################################################
# # Analyze and model an infant’s birth weight
# #########################################################################
# 
# #############
# ## Challenge:
# #############
# Certain factors contribute to the health of a newborn baby. One such health measure is birth weight.
# Countless studies have identified factors, both preventative and hereditary, that lead to low birth weight.
# Your team has been hired as public health consultants to analyze and model an infant’s birth weight based on such characteristics.
####################################################################################################################################


# ###############################
# ## Birthweight_data_dictionary:
# ###############################

# | variable | label   | description                     |
# 
# |----------|---------|---------------------------------|
# 
# | 1        | mage    | mother's age                    |
# 
# | 2        | meduc   | mother's educ                   |
# 
# | 3        | monpre  | month prenatal care began       |
# 
# | 4        | npvis   | total number of prenatal visits |
# 
# | 5        | fage    | father's age, years             |
# 
# | 6        | feduc   | father's educ, years            |
# 
# | 7        | omaps   | one minute apgar score          |
# 
# | 8        | fmaps   | five minute apgar score         |
# 
# | 9       | cigs    | avg cigarettes per day          |
# 
# | 10       | drink   | avg drinks per week             |
# 
# | 11       | male    | 1 if baby male                  |
# 
# | 12       | mwhte   | 1 if mother white               |
# 
# | 13       | mblck   | 1 if mother black               |
# 
# | 14       | moth    | 1 if mother is other            |
# 
# | 15       | fwhte   | 1 if father white               |
# 
# | 16       | fblck   | 1 if father black               |
# 
# | 17       | foth    | 1 if father is other            |
# 
# | 18       | bwght   | birthweight, grams              |
# ########################################################




# ####################################
# # Importing the Libraries & the Data
######################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


file = 'birthweight_feature_set.xlsx'
bw = pd.read_excel(file)




# #####################################
# # Exploratory analysis of the dataset
#######################################

# Column names
bw.columns


# Displaying the first rows of the DataFrame
print(bw.head())


# Dimensions of the DataFrame
bw.shape


# Information about each variable
bw.info()


# Descriptive statistics
bw.describe().round(2)




# #############################
# # Dealing with missing values
###############################

# Total of missing values
print(bw
      .isnull()
      .sum()
      .sum()
)

# Missing values per column
print(bw
      .isnull()
      .sum()
)


# ### Columns with missing values:
# * meduc:     3
# * npvis:     3
# * feduc:     7

# ### Let's fill the missing values:

# Now than we saw that the missing values in this dataset is not "big deal" because we have less than 5% of rows compromissed.
# Here is where we nee to decide if we just dropp the missing values or we apply a technoque to fill in the values.
# In this case because are a few is almost the same dropp them or fill with the median.

# If you have more missing values you can fill it with:
# * Median for distributions with skewness (right or left)
# * Mean with normal (or almost) distributions (in my case I preffer to always use the median because there's almost no difference if is normal distributed.
# * Linear regression to predict the values using another variable correlated to this variables with missing values (here we need to be carefull, thinking in the future, with Multicollinearity (https://en.wikipedia.org/wiki/Multicollinearity ).
# * Another one, but more difficult, is to find other obserbations inside of the dataset with similar characteristics of the ones with missing values and fill the NaS with the data from this other observations.

### Here we will use the "median" to fill in the missing values ###


# Let's first create a new df with with the imputation
bw_m = bw

for col in bw_m.columns:
    if (bw_m[col].isnull().any()) == True:
        bw_m[col] = bw_m[col].fillna(bw_m[col].median())


# Now let's check the missing values were correctly imputed
bw_m.info()

print(bw_m.describe()
      .round(2))

print(bw_m.isnull()
      .sum()
      .sum())


#######################
# ### Analysis of our objective variable (dependent): Birth Weight (bwght)
# 
# Here we will see how is this variable distributed and other aspects of it such as:
# * Descriptive statistics summary
# * Distribution
# 
# Let's see how is our variable like

#######################
# ### Summary of Descriptive Statistics
# Here we have view of the shape and description of our predictive (dependent) variable.

# Let's see the summary of the descriptive statistics
bw_m['bwght'].describe()


# Here we can see that the minimun value is greater than 0, meaning that there is no values that will destroy the model when created. Also, we can see that the min value is really low and the normal mid values (the IQR) the values rage between 2916 Kgs and 3759 that is a normal weight for birth weights: **a mean of 3.5 kilograms, though the range of normal is between 2.5 kilograms and 4.5 kilograms**. (source: https://en.wikipedia.org/wiki/Birth_weight )*
# 
# After looking at this data, we can see that a good weight at birth is a really important factor of health in life, like the source said: *"to show links between birth weight and later-life conditions, including **diabetes, obesity, tobacco smoking, and intelligence**. Low birth weight is associated with **neonatal infection** and **infant mortality**" (source: https://en.wikipedia.org/wiki/Birth_weight ).*
# 
# As wee can see also the MAX value is normal, 4933Kgs, so now let's see how many low values we have, because this cases can be a reflection of a low quality of health.


######################
# ### Let's sort the values of Birth Weight from ascending (low to bigger)
# Here we sort the values and display only the first 20 results.
bw_m['bwght'].sort_values(ascending = True).head(20)


# Here we can see that the first 15 values are below the min value of the normal range for a baby to be healthy. We can analize this observations to see what's happening on the other variables in order to try to predict the low brthweight.*

#######################
# ### Distribution of the objective variable
# Now let's see the distribution of the Birth Weight variable, to understand its behaviour
sns.distplot(bw_m['bwght'])


# We can see a light left Skewness, but is not strong.




########################
# # Correlation Analysis
########################


# Now that we've done the analysis on the varibles and some web exploration, let's do some data correlation between the variables to see what else we can find and also to see corr() between the "birth weight" and some other variables.


# Here we do the correlation
bw_m_corr = bw_m.corr()

# now let's do the graph of the heatmap
fig, ax=plt.subplots(figsize=(10,10))
sns.set(font_scale=2)
sns.heatmap(bw_m_corr,
            cmap = 'Blues',
            square = True,
            annot = False,
            linecolor = 'black',
            linewidths = 0.5)

plt.savefig('correlation_matrix_all_var')
plt.show()


# ###From the visual approach we can see that there are some interesting correlations:
# * 1) Birth Weight and cigs = Negative
# * 2) Birth Weight and drink = Negative
# * 3) Birth Weight and mage = Negative
# * 4) Birth Weight and fage = Negative
# * 5) Moth (mother other) and feduc = Positive
# * 6) foth and feduc = Positive
# * 7) fwhite and feduc = Negative

######################
# ## Strong Numerical Correlations
# Let's take the strongest correlation (positives & negatives) related to "bwght" (objective variable)

# Let's print ascending first
bw_m_corr['bwght'].sort_values(ascending=True).head(10)


# Here we can see that there are 2 strong correlated variables that are related to have a low performance in terms of Birth Weight, this ones are "drink" and "cigs". Later we will try to change the values of "cigs" to make them binary to see if there more chance of a bigger corr()

######################
# #### Now let's try with the positive correlated variables.

# Let's print ascending= False
bw_m_corr['bwght'].sort_values(ascending=False).head(10)

# There's nothing with a strong positive correlation with bwght

#####################
# ### Other Variables Related With Health
# Let's try now with other variable related to the health of a born baby: "omaps" that is the Apgar Test

# Let's print ascending first
bw_m_corr['omaps'].sort_values(ascending=True).head(10)

# Let's print ascending= False
bw_m_corr['omaps'].sort_values(ascending=False).head(10)

# As we can see here there are no strong correlations using this variable.

####################
# ## Correlations Graphs

# Now let's plot all the correlations that we think are interesting after looking the heatmap and the numerical correlations, so we can have a graphical taste of them.
sns.set()
cols = ['bwght', 'drink', 'cigs', 'omaps', 'mage', 'fage', 'moth', 'feduc', 'foth', 'fwhte']
sns.pairplot(bw_m[cols], height= 2.5)
plt.show();

#####################
# # Numerical Variables Distribution

# Let's plot all the variables ditributions to have a graphical idea of the distribution of all our variables
bw_m.hist(figsize=(16, 20), bins=50, xlabelsize=12, ylabelsize=12)

# Here we can see graphicaly the binary variables like mwhte, moth, mblck, etc. Also we can see is how the age of the fathers is not so high and the weight is mostly between the normal values, what means young fathers (mostly below 35-40) here are the most common as the normal weight of the babies***



#####################
# # Outliers Analysis
#####################

# Let's take a look on the outliers before going into Model Land.
# 
# Let's make some boxplots in order to see the outliers in the different variables.

# Let's make a loop in order to graph all the boxplots
for col in bw_m:
    sns.set()
    plt.figure(figsize = (7, 3))
    ax = sns.boxplot(x=bw_m[col], data=bw_m)
    plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
    plt.xticks(rotation=45)


# Here was really interest the outliers of the following variables:
# * mage
# * fage
# * drink
# * bwght
 
##########################
# Now let's see the values for this variables
bw_m[['mage', 'fage', 'drink', 'bwght']].describe()


#Now let's graph the scater plots of the variables with interesting outliers**
sns.set()
cols = ['bwght', 'drink', 'mage', 'fage']
sns.pairplot(bw_m[cols], height= 2.5)
plt.show();

# Here we can see something really interesting, take a look to the outliers value in the scater plot relating bwght vs mage and you can see a couple of outliers with a low birth weight.***


# let's go there and see which are this values and what is happening there
bw_m_outliers = bw_m.sort_values(ascending = True, by = 'bwght').head(4)
print(bw_m_outliers)
bw_m_outliers.describe()



bw_m_outliers_corr = bw_m_outliers.corr()
# now let's do the graph of the heatmap
fig, ax=plt.subplots(figsize=(10,10))
sns.set(font_scale=2)
sns.heatmap(bw_m_outliers_corr,
            cmap = 'Blues',
            square = True,
            annot = False,
            linecolor = 'black',
            linewidths = 0.5)

plt.savefig('correlation_matrix_all_var')
plt.show()


# ***Really interesting results:***
# Here we find that not only this babies have low weight, but also they have a low Apgar Score (mean omaps= 4.25), and also they have another thing related that is the age of the mother and father, in hthis cases the AVG Mother Age is 66.75 (the highest value in age is 71 this is super rare), and the AVG Father Age is 54.25, and also in terms of bad habits they have all, the AVG of cigs is 22.5 per day and drinks is 10.50 per week, tgis value are between the highest of these variables.




############################
### Creating a new data set
############################

# After our analysis we can see cleary that we need to make some changes in our dataset, such as making some dummy variables:
# * mage outliers (mother outliers)
# * Birth Weight outliers
# * fage outliers (fathers age)
# * AVG cigarettes
# * AVG drinking

# All this variables we will gonna flag all the higher and lower outliers


#Outlier Imputation
bwght_hi = 4001
bwght_lo = 2499

mage_hi = 55
mage_lo = 14

meduc_hi = 17 
meduc_lo = 8

fage_hi = 60
fage_lo = 23

feduc_hi = 17 
feduc_lo = 8

monpre_hi = 4

npvis_hi = 15
npvis_lo = 5

omaps_hi = 10
omaps_lo = 7

fmaps_hi = 10
fmaps_lo = 7

cigs_hi = 20

drink_hi = 10


########################
# Creating Outlier Flags
########################

# Building loops for outlier imputation
# Let's first create a nwew data set where we are going to put the new dummy variables
# to use later in our model creation.
bw_m2 = pd.DataFrame.copy(bw_m)


########################
# Birthweight

bw_m2['out_bwght'] = 0


for val in enumerate(bw_m2.loc[ : , 'bwght']):
    
    if val[1] <= bwght_lo:
        bw_m2.loc[val[0], 'out_bwght'] = -1
        
for val in enumerate(bw_m2.loc[ : , 'bwght']):    
    
    if val[1] >= bwght_hi:
        bw_m2.loc[val[0], 'out_bwght'] = 1

        
########################
# Mother's Age

bw_m2['out_mage'] = 0


for val in enumerate(bw_m2.loc[ : , 'mage']):
    
    if val[1] <= mage_lo:
        bw_m2.loc[val[0], 'out_mage'] = -1
        
for val in enumerate(bw_m2.loc[ : , 'mage']):    
    
    if val[1] >= mage_hi:
        bw_m2.loc[val[0], 'out_mage'] = 1
        
        
########################
# Father's Age

bw_m2['out_fage'] = 0


for val in enumerate(bw_m2.loc[ : , 'fage']):
    
    if val[1] <= fage_lo:
        bw_m2.loc[val[0], 'out_fage'] = -1


for val in enumerate(bw_m2.loc[ : , 'fage']):
    
    if val[1] <= fage_hi:
        bw_m2.loc[val[0], 'out_fage'] = 1

        
########################
# Average Cigertte

bw_m2['out_cigs'] = 0

for val in enumerate(bw_m2.loc[ : , 'cigs']):
    
    if val[1] >= cigs_hi:
        bw_m2.loc[val[0], 'out_cigs'] = 1


########################
# Average Drinking 
        
bw_m2['out_drink'] = 0


for val in enumerate(bw_m2.loc[ : , 'drink']):
    
    if val[1] >= drink_hi:
        bw_m2.loc[val[0], 'out_drink'] = 1


# let's see the new things inside of our data!
bw_m2.columns

#############
# Explanation
# As we can see when we print the name of the columns, we can now see that our new dummy varialbes have been created. So, now, we can use this to apply in our models, to have a better accuracy or better insights.

#################
# Some more graphs to use in our finanl report

bw_m3 = pd.DataFrame.copy(bw)

# Cigs
bw_m3['if_cigs']=0
for val in enumerate(bw_m3.loc[ : , 'cigs']):
    if val[1] == 0:
        bw_m3.loc[val[0], 'if_cigs'] = 'No'
    else:
        bw_m3.loc[val[0], 'if_cigs'] = 'Yes'
bw_m3['bwght'].mean()

sns.lmplot(x="cigs", y="bwght", data=bw_m3,
           fit_reg=True,scatter=True,hue='if_cigs',palette='pastel')
plt.axhline(3334,color='green')


# Drink
bw_m3['if_drink']=0
for val in enumerate(bw_m3.loc[ : , 'drink']):
    if val[1] == 0:
        bw_m3.loc[val[0], 'if_drink'] = 'No'
    else:
        bw_m3.loc[val[0], 'if_drink'] = 'Yes'

sns.lmplot(x="drink", y="bwght", data=bw_m3,
           fit_reg=True,scatter=True,palette='pastel',hue='if_drink')
plt.axhline(bw_m3['bwght'].quantile(0.5),color='red')


# Now the last graph to divide in different groups the corr() graph
# between mage and bwght
age=bw_m3.loc[:,['mage','fage','bwght']]
hi_age=age[(age['mage']>=35)&(age['fage']>=35)]
fh_ml=age[(age['mage']<35)&(age['fage']>=35)]
fl_mh=age[(age['mage']>=35)&(age['fage']<35)]
fl_ml=age[(age['mage']<35)&(age['fage']<35)]


# Segment the group
hi_age.drop(['mage','fage'],axis=1)
hi=hi_age['bwght'].tolist()
age['group']=0
for val in enumerate (age.loc[:,'bwght']):
    if val[1] in hi:
        age.loc[val[0],'group']='hi'
        
fl_ml.drop(['mage','fage'],axis=1)
lo=fl_ml['bwght'].tolist()
for val in enumerate (age.loc[:,'bwght']):
    if val[1] in lo:
        age.loc[val[0],'group']='Normal'


fh_ml.drop(['mage','fage'],axis=1)
fhml=fh_ml['bwght'].tolist()
for val in enumerate (age.loc[:,'bwght']):
    if val[1] in fhml:
        age.loc[val[0],'group']='fhml'


fl_mh.drop(['mage','fage'],axis=1)
flmh=fl_mh['bwght'].tolist()
for val in enumerate (age.loc[:,'bwght']):
    if val[1] in flmh:
        age.loc[val[0],'group']='flmh'


#####Comprison
sns.FacetGrid(age, hue="group", height=5) \
   .map(plt.scatter, "mage", "bwght") \
   .add_legend()
plt.axhline(3000,color='purple')
bw['bwght'].mean()

plt.axhline(age['bwght'].mean(),color='purple',label='Mean of BW')

plt.savefig('Birthweight Group.png')


########
# Export

# Let's export our exploratory analisis to an excel file to have a firewall here in case we want to go back to this file in the future.
bw_m2.to_excel('Birthweight_explored.xlsx')


###############################################################################################################
# ### Interesting sources for the data exploratory analysis:
# * https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python/notebook
# * https://www.kaggle.com/ekami66/detailed-exploratory-data-analysis-with-python
# * https://kite.com/blog/python/data-analysis-visualization-python
# * https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch09_Analyzing_Wine_Types_and_Quality/Exploratory%20Data%20Analysis.ipynb
# * https://medium.com/ibm-data-science-experience/markdown-for-jupyter-notebooks-cheatsheet-386c05aeebed
###############################################################################################################




###############################
## Modeling - Machine Learning
###############################

# Now let's go and have some fun trying to predict our objective variable (Birth Weight) using different models and approaches possibly changing some variables values.

# Because we are dealing with a continuos variable, we are going to use the following models:
# * OLS Regression
# * KNN (K-Nearest Neighbors)
# * Decision Tree Regressor
 
#####################
# ### First let's import the libraries needed

# Libraries needed to do the modeling
import statsmodels.formula.api as smf # regression modeling




######################################
## OLS Regression - 1st Model Applied
######################################

# As a fisrt approach let's use all the variables just to have a look at the p-values and the variables than don't make sense to remainin the model.

# Also we are going to use the dataset "bw_m2" that is the one that has the dummy variables and the outliers spilt in the mother age, so then we can get a better approach because this outliers predict better birthweight as they are extreme cases.

## Let's bring the data set saved from the exploratory analisys
import pandas as pd
bw_m2 = pd.read_excel('Birthweight_explored.xlsx')

# Let's create the model using all the other variables as they are.
ols_full = smf.ols(formula = """bwght ~ bw_m2['fmaps'] +
                                    bw_m2['omaps'] +
                                    bw_m2['out_fage'] +
                                    bw_m2['feduc'] +
                                    bw_m2['mblck'] +
                                    bw_m2['fblck'] +
                                    bw_m2['male'] +
                                    bw_m2['meduc'] +
                                    bw_m2['npvis'] +
                                    bw_m2['moth'] +
                                    bw_m2['fwhte'] +
                                    bw_m2['monpre'] +
                                    bw_m2['foth'] +
                                    bw_m2['mwhte'] +
                                    bw_m2['out_cigs'] +
                                    bw_m2['out_mage'] +
                                    bw_m2['out_drink'] +
                                    bw_m2['fage'] +
                                    bw_m2['mage'] +
                                    bw_m2['cigs'] +
                                    bw_m2['drink']
                                    """,
                   data = bw_m2)


# Fitting Results
results_ols_full = ols_full.fit()



# Printing Summary Statistics
print(results_ols_full.summary())



print(f"""
Summary Statistics:
R-Squared:          {results_ols_full.rsquared.round(3)}
Adjusted R-Squared: {results_ols_full.rsquared_adj.round(3)}
""")


#############
# ### Results

# Here we see that we get an R-Squared = 0.744, is good to be the first approach without touching the data.<br>
# We can see a lot of variables with big p-values and some variables that doesn't make sense like the omaps and fmaps because they meassure the baby when he is outside of the mother and not before he is born.


###################
# ### Deleting Some Variables From the Model

# After seeing the p-values and trying different approaches (run several times the model) let's reamain with the following variables:
# * monpre
# * meduc
# * cigs
# * drink
# * npvis
# * out_mage

##################
# Let rid this variables an re-build the model and see the results.


# Let's create the model using all the other variables as they are.
bw_m_ols = bw_m
ols_significant = smf.ols(formula = """bwght ~ 
                                    bw_m2['monpre'] +
                                    bw_m2['meduc'] +
                                    bw_m2['cigs'] +
                                    bw_m2['drink'] +
                                    bw_m2['npvis'] +
                                    bw_m2['out_mage']
                                           """,
                         data = bw_m2)


# Fitting Results
results_ols_significant = ols_significant.fit()



# Printing Summary Statistics
print(results_ols_significant.summary())



print(f"""
Summary Statistics:
R-Squared:          {results_ols_significant.rsquared.round(3)}
Adjusted R-Squared: {results_ols_significant.rsquared_adj.round(3)}
""")


# ###########
# ### Results

# Now we get a lower R-Squared = 0.725, but our model used much less variables (6 variables) and makes more sense in order to predict. And without making a split / test of the data. Let's do that next.<br>




#####################
### Let's save this
#####################

bw_m.to_excel('BirthWeight_OLS.xlsx')

# With this model we get an R-Squared = 0.725



# So let's go and try another model: KNN with also with this dataset "bw_m2", that has the modified variables with the dummy one.
# And also next we will apply the OLS with a split in the data (training / test) and compare it with KNN.Is always importan to SPLIT the DATA, that's why our fisrt model in OLS will probaly be more accurate




##################################################################################
### KNN Model(K-Nearest Neighbors) & OLS Comparisson (Data Split: Test & Training)
##################################################################################

# Now it's time to KNN to shine! I know you can do a great job predicting! And let's compare it with OLS splitting the data
# Let's import the libraries needed here:


# Importing new libraries
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import statsmodels.formula.api as smf # regression modeling
import sklearn.metrics # more metrics for model performance evaluation

#####################
# Set up the Model for OLS
# Let's create the model below, but first let's prepare the data, let's split, create our train and test sets. Here we prepare the best fit of OLS matching the better mix of variables.


# Here we will do all the preparaion before runing the model
# Let's rid the column that we are wanting to predict and the ones that we already know are not good for our model
bw_data   = bw_m2.drop(['bwght',
                          'omaps',
                          'fmaps',],
                           axis = 1)


# Seting up the target we want to predict
bw_target = bw_m2.loc[:, 'bwght']


# Let's now split the data to Train and then Test our model
X_train, X_test, y_train, y_test = train_test_split(
            bw_data,
            bw_target,
            test_size = 0.1,
            random_state = 508)


# Training set 
print(X_train.shape)
print(y_train.shape)


# Testing set
print(X_test.shape)
print(y_test.shape)


########################
# Now let's create the model - But first let's ring our optimal model OLS to compare
# After spliting the data we create the model in order to run it first without the binary variables.<br>
# Here we bring the ols model but we are going to use it spliting the data, we will probably get different results because we have less data in the training, which is the one that we use here.


# We need to merge our X_train and y_train sets so that they can be
# used in statsmodels
bw_train = pd.concat([X_train, y_train], axis = 1)

# Let's pull in the optimal model from before, only this time on the training set
knn_significant = smf.ols(formula = """bwght ~
                                            bw_train['meduc'] +
                                            bw_train['cigs'] +
                                            bw_train['drink'] +
                                            bw_train['out_fage'] +
                                            bw_train['out_mage']
                                            """,
                            data = bw_train)


# Fitting Results
knn_results = knn_significant.fit()


# Printing Summary Statistics
print(knn_results.summary())


# Notice how the R-Squared is bigger, now is 0.736 (compared to the previous model).
# This is because we are dealing with less data, and there may not be enough observations to establish the same trend than before. But next we are going to use the TEST score to see the real accuracy of the model




##############################################
### Applying Our Optimal Model in scikit-learn
##############################################

# Now that we have selected our variables, our next step is to prepare 
# them in scikit-learn so that we can see how they predict on new data.

# Preparing a DataFrame based the the analysis above
bw_data   = bw_m2.loc[ : , ['meduc',
                           'cigs',
                           'drink',
                            'out_fage',
                           'out_mage']]


# Preparing the target variable
bw_target = bw_m2.loc[:, 'bwght']


# Now that we have a new set of X_variables, we need to run train/test
# split again
X_train, X_test, y_train, y_test = train_test_split(
            bw_data,
            bw_target,
            test_size = 0.1,
            random_state = 508)




#############################################################################
### Using KNN  On Our Optimal Model (same code as our previous script on KNN)
#############################################################################

# Import the libraries
from sklearn.linear_model import LinearRegression


# Prepping the Model
lr = LinearRegression()


# Fitting the model
lr_fit = lr.fit(X_train, y_train)


# Predictions
lr_pred = lr_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', lr.score(X_train, y_train).round(2))
print('Testing Score:', lr.score(X_test, y_test).round(2))

# Here we can see the results of the OLS model using the training and testing data:**
# * Training Score 0.74
# * Testing Score: 0.6916


# And when we compare to the results of the KNN model we see an accuracy:
# * 0.64 as we are going to see below

# Exact loop as before
training_accuracy = []
test_accuracy = []


neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)

########################
# ### Partial conclusions
# Here we need to look at the performance of the test accuracy, we can see that the accuracy goes down if we put more than k=15.<br>
# Using k=15 we can get an accuracy around 50%, let's continue and see the final accuracy below.<br>
# **Let's try now changing the k=15**


# Building a model with k = 15
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 15)



# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)



# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)



# The score is directly comparable to R-Square
print(y_score_knn_optimal)



# Generating Predictions based on the optimal KNN model
knn_reg_optimal_pred = knn_reg_fit.predict(X_test)

# #### ***Here we get accuracy of 64%. Lets continue with the analysis and comapre this model with the OLS one.***


###############
### Conclusion

# #### ***So the conclusion is that here is better to use the OLS model in order to have a better accuracy aproach.***
# #### ***And also it is important to highlight that the accuracy of the model using the TEST set is below the TRAINING, which means how the model act when presenting new data. Here is a point to run a cross-validation test to get more info.***




##########################################
### Storing Model Predictions and Summary
##########################################

# We can store our predictions as a dictionary (first let's save all the preictions).
model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'KNN_Predicted': knn_reg_optimal_pred,
                                     'OLS_Predicted': lr_pred})


model_predictions_df.to_excel("BW_Model_KNN_Predictions.xlsx")

##############
# Now let's save only the predictions of our best model
model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'OLS_Predicted': lr_pred})


model_predictions_df.to_excel("BW_Model_OLS_Predictions.xlsx")




#########################
# ## Decision Trees Model
#########################

# Let's now give a chance to decision trees model, we know that the accuracy of this model is not the best but is a good option to stick to our business problem and to have a graphical approach.


# Libraries importation
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects
from sklearn.model_selection import cross_val_score # k-folds cross validation


###################
# Train/Test Splits

# Preparing a DataFrame based the the analysis above
bw_data   = bw_m.loc[ : , ['mage',
                           'meduc',
                           'monpre',
                           'npvis',
                           'cigs',
                           'drink',
                           'male',
                           'mwhte',
                           'mblck',
                           'moth',
                           'fwhte',
                           'fblck',
                           'foth']]


# Preparing the target variable
bw_target = bw_m.loc[:, 'bwght']


# Now that we have a new set of X_variables, we need to run train/test
# split again

X_train, X_test, y_train, y_test = train_test_split(
            bw_data,
            bw_target,
            test_size = 0.1,
            random_state = 508)

####################
# ### Decision Trees

# Now let's go to our star of the section. A star is born? Future Lady Gaga?<br>
# Let's see...

# Notice how all of our modeling techniques in scikit-learn follow the same steps:
# * create a model object
# * fit data to the model object
# * predict on new data
# * score

# If you can master these four steps, any modeling technique in scikit-learn is at your disposal.

# Let's start by building a full tree.
tree_full = DecisionTreeRegressor(random_state = 508)
tree_full.fit(X_train, y_train)

print('Training Score', tree_full.score(X_train, y_train).round(4))
print('Testing Score:', tree_full.score(X_test, y_test).round(4))

############################
# ### Analyzing the results.
# Notice that the training score is extremely high. This is because the tree kept growing until the response variable was sorted as much as possible. In the real world we do not want a tree this big because:
# * it will predict poorly on new data, as observed above
# * it is too detailed to gain meaningful insights

# Displaying this tree would be a HUGE mistake. Let's manage the size of our tree through pruning.

tree_2 = DecisionTreeRegressor(criterion = 'mse',
                                     min_samples_leaf = 10,
                                     max_depth = 3,
                                     random_state = 508)

tree_2_fit = tree_2.fit(X_train, y_train)


print('Training Score', tree_2.score(X_train, y_train).round(4))
print('Testing Score:', tree_2.score(X_test, y_test).round(4))


###########
### Scores
# Here we can see hoe the score of the tree went down from 1.00 to 0.64, is not a bad result but our other model are more accurate.<br>
# Here we change the size of the tree, we try different min_samples_leaf and max_depth in order to reach the best accuracy, but most, because now we have below a better graph to show and makes sense.


dot_data = StringIO()


export_graphviz(decision_tree = tree_2_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = bw_data.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png(),
      height = 500,
      width = 800)


# Here we can see a chart that has a reasonable size and we can show some insights strongly related to our business problem, but the accuracy is not better than our previously models like OLS or KNN that we reached better scores in terms of accuracy

# Saving the tree visualization
graph.write_png("BW_Decision_Tree.png")


# While this tree could be much deeper, at its current depth it can be overwhelming for some audiences, especially when trying to develop insights. To assist with this, a common approach summarize using feature importance. This technique rates each feature in terms of how much it was used in developing the tree.

print(tree_2.feature_importances_)


#Defining a function to visualize feature importance
def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_2_Feature_Importance.png')
########################


plot_feature_importances(tree_2,
                         train = X_train,
                         export = True)



# Let's plot feature importance on the full tree.
plot_feature_importances(tree_full,
                         train = X_train,
                         export = False)


######################################################
############ Final Conclusions & Notes ###############
######################################################
# ### Conclusion of the Decision Tree Model
# Here we can see (First Plot) that in the model with 10 leafs (the second one that we adjust, not the full one) that the model mostly use drink in order to predict and is used as a breakpoint and second the cigs and 3rd the Mother Age.<br>
# In the second plot (the one with full leafs, automatic) the model is defined 1st by drink, 2nd cigs and 3rd Mother Age, but is good to see that also takes in account npvis as 4th variable, here we can see why is important to keep with this variable in the previous model (OLS and KNN)




# ## Final Conclusion of Modeling
# Now that we have done all the models, we can summaryze the results of every model:
# <br>
# 
# OLS Model:
# * R-Squared: 0.74<br>
#
# And from the Trainig and Spliting sets we get the follolwing results:
# * Training Score 0.74
# * Testing Score: 0.69
# * We can see there's a gap between the two, and also the model is approaching 70% of accuracy, is not a bad model and we took care of the variables that are too big in terms of p-value. We could gain 1% of accuracy in the prediction but that will be having 2 variables with p-values really high.


# KNN Model:
# * Accuracy: 0.50
#
# Decision Tree:
# * Training Score 0.64
# * Testing Score: 0.53

# ***After looking at this we can see that the OLS model has the best predictor accuracy***




############################################################################################
#                                                                                          #
####################################### END ################################################
#                                                                                          #
############################################################################################


