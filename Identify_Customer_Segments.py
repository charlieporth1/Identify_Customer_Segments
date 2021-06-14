#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[1]:


get_ipython().system('pip install wheel asyncio')
#!pip install pandas seaborn pycocotools scipy scikit-learn==0.19.2  matplotlib
#!pip install numpy==1.20.2 


# In[2]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

np.seterr(all="ignore")

float
bool 
int
np.float64
np.float
np.int
np.bool_
np.bool

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[3]:


# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')
# Load in the feature summary file.
feat_info = pd.read_table('AZDIAS_Feature_Summary.csv',sep=';')


# In[4]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).
feat_info.head()
azdias.head()


# In[5]:


feat_info.describe()
azdias.describe()


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[6]:


# Identify missing or unknown data values and convert them to NaNs.

def find_missing_or_uknown(fi, az): 
    for index, MoU in zip(range(len(fi)), fi['missing_or_unknown']):
        att = fi.iloc[index]['attribute'] 
        MoU = MoU.replace('[', '').replace(']', '').split(",")
        MoU_values = [int(value) if (value != 'X' and value != 'XX' and value != '') else value for value in MoU]
        if MoU_values != ['']:
            az[att].replace(to_replace = MoU_values, value = np.nan, regex = True, inplace = True ) 
            
    return az
print('naturally missing {0}'.format(azdias.isnull().sum().sum()))

azdias = find_missing_or_uknown(feat_info, azdias)

print('number of missing values {0}'.format(azdias.isnull().sum().sum()))
        


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[7]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.
col_per_mis = (azdias.isnull().sum()/len(azdias))*100


# In[8]:


# Investigate patterns in the amount of missing data in each column.

plt.hist(col_per_mis, bins=100)
plt.ylabel('Col')
plt.xlabel('% Mis')
plt.show()


# In[9]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)
def drop_less():
    col_per_mis_20 = col_per_mis[col_per_mis>20]
    drop_coloumns = col_per_mis_20.index.tolist()
    return drop_coloumns
drop_coloumns = drop_less()
print('20 percent missing data == drop {0}'.format(drop_coloumns))


# In[10]:


azdias = azdias.drop(drop_coloumns, axis=1)


# In[11]:


azdias.head()


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# (Double click this cell and replace this text with your own text, reporting your observations regarding the amount of missing data in each column. Are there any patterns in missing values? Which columns were removed from the dataset?)
# 
# Columns with less than 20% data are dropped 
# 
# Are removed from the data set
# ['AGER_TYP', 'GEBURTSJAHR', 'TITEL_KZ', 'ALTER_HH', 'KK_KUNDENTYP', 'KBA05_BAUMAX']

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[12]:


# How much data is missing in each row of the dataset?

missing_per_row = azdias.isnull().sum(axis=1)
plt.hist(missing_per_row, bins=100)
plt.ylabel('Rows')
plt.xlabel('Missing Values')
plt.show()


# In[13]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
azdias_subset_1 = azdias[azdias.isnull().sum(axis=1) <= 3]
azdias_subset_2 = azdias[azdias.isnull().sum(axis=1) > 3]

zero_missing_col = col_per_mis[col_per_mis==0].index.tolist()
compare_col = zero_missing_col[:6]


# In[14]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.

def checkmissingnum(azdias_subset_1, azdias_subset_2, random_index):
    
    columnname = azdias_subset_2.columns[random_index]

    fig,axes = plt.subplots(nrows = 1, ncols = 2, figsize = ( 20 , 2 ))
    titlestring = 'feature ' + columnname
    plt.title(titlestring)
    sns.countplot( x = columnname, data = azdias_subset_1, ax = axes[ 0 ] )
    sns.countplot( x = columnname, data = azdias_subset_2 , ax = axes[ 1 ] )
    
    pass

import random as random
for i in range(0,10):
    random_index = random.randrange(39)
    checkmissingnum(azdias_subset_1, azdias_subset_2, random_index)


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# (Double-click this cell and replace this text with your own text, reporting your observations regarding missing data in rows. Are the data with lots of missing values are qualitatively different from data with few or no missing values?)
# 
# You can use seaborn's countplot() function to create a bar chart of code frequencies and matplotlib's subplot() function to put bar charts for the two subplots side by side.
# To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[15]:


# How many features are there of each data type?

features_types = []
features = list(azdias_subset_1.columns)
feat_info_clean = feat_info[feat_info['attribute'].isin(features)]
data_type_count = feat_info_clean['type'].value_counts()

for i in range(len(data_type_count)):
    print('There are {} {} features.'.format(data_type_count[i], data_type_count.index[i]))
    features_types += [data_type_count.index[i]]
    
print(features_types)


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[16]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
def feature_encoder(feat_info_clean): 
    cat_features = feat_info_clean[feat_info_clean["type"] == "categorical"]["attribute"]

    binary_feature = []
    multi_level_feature = []
    for feature in cat_features:
        if (len(azdias_subset_1[feature].unique())==2):
            binary_feature.append(feature)
        elif (len(azdias_subset_1[feature].unique())>2):
            multi_level_feature.append(feature)
    return binary_feature, multi_level_feature
        
binary_feature, multi_level_feature = feature_encoder(feat_info_clean) 

print('Binary features are: {}'.format(binary_feature))
print('Multi-level features are: {}'.format(multi_level_feature))


# In[17]:


dataset_features = {}
for data in features_types:
    dataset_features[data] = feat_info[ feat_info['type'] == data ]['attribute']


# In[18]:


for feature in binary_feature:
    print('Unique values for {} are {}'.format(feature, azdias_subset_1[feature].unique()))

replace_values = {'W': 0, 'O': 1}

azdias_clean = azdias_subset_1.replace({'OST_WEST_KZ':replace_values}) 

def get_drop(azdias_clean, multi_level_feature):
    for feature in multi_level_feature:
        azdias_clean = azdias_clean.drop(feature, axis=1)
        
    return azdias_clean


azdias_clean = get_drop(azdias_clean, multi_level_feature)


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding categorical features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)
# 
# First I added the mixed values
# Then I added the categorical/multi-level features

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[19]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
def azdias_data_clean_1(azdias_data_clean): 
    azdias_data_clean['DECADE'] = azdias_data_clean['PRAEGENDE_JUGENDJAHRE']
    azdias_data_clean['MOVEMENT'] = azdias_data_clean['PRAEGENDE_JUGENDJAHRE']

    decade_dict = {1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:3, 8:4, 9:4, 10:5, 11:5, 12:5, 13:5, 14:6, 15:6}
    movement_dict = {1:1, 2:0, 3:1, 4:0, 5:1, 6:0, 7:0, 8:1, 9:0, 10:1, 11:0, 12:1, 13:0, 14:1, 15:0}

    azdias_data_clean['DECADE'].replace(decade_dict, inplace=True)
    azdias_data_clean['MOVEMENT'].replace(movement_dict, inplace=True)
    return azdias_data_clean


azdias_clean = azdias_data_clean_1(azdias_clean)


# In[20]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.
def azdias_data_clean_2(azdias_data_clean): 
    azdias_data_clean['WEALTH'] = azdias_data_clean['CAMEO_INTL_2015']
    azdias_data_clean['LIFE_STAGE'] = azdias_data_clean['CAMEO_INTL_2015']

    wealth_dict = {'11':1, '12':1, '13':1, '14':1, '15':1, '21':2, '22':2, '23':2, '24':2, '25':2,
               '31':3, '32':3, '33':3, '34':3, '35':3, '41':4, '42':4, '43':4, '44':4, '45':4,
               '51':5, '52':5, '53':5, '54':5, '55':5}

    life_stage_dict = {'11':1, '12':2, '13':3, '14':4, '15':5, '21':1, '22':2, '23':3, '24':4, '25':5,
                   '31':1, '32':2, '33':3, '34':4, '35':5, '41':1, '42':2, '43':3, '44':4, '45':5,
                   '51':1, '52':2, '53':3, '54':4, '55':5}

    azdias_data_clean['WEALTH'].replace(wealth_dict, inplace=True)
    azdias_data_clean['LIFE_STAGE'].replace(life_stage_dict, inplace=True)

    return azdias_data_clean

azdias_clean = azdias_data_clean_2(azdias_clean)


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding mixed-value features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)
# 
# I did this
# 
# Created two new feature columns, and copied values from the initial mixed-value feature column
# Created two dictionaries to map intial feature values to new feature values
# Performed a replace operation using the dictionaries on the new feature columns
# I decided to drop the other four mixed-values features.

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[21]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)
def mixed_feature_encoder(azdias_clean):
    mixed_features = feat_info_clean[feat_info_clean["type"] == "mixed"]["attribute"]
    for feature in mixed_features:
        azdias_clean.drop(feature, axis=1, inplace=True)

    return azdias_clean
        
azdias_clean = mixed_feature_encoder(azdias_clean)


# In[22]:


# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.
azdias_clean.head()


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[23]:


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    df_clean = find_missing_or_uknown(feat_info, df)
    
    for col in df.columns:
        df_clean = df_clean.replace({col: ['XX', 'X']}, np.nan)
        
    # drop columns with more than 20% missing values
    drop_coloumns = drop_less()
    df_clean = df_clean.drop(drop_coloumns, axis=1)
    
    # drop rows with more than 3 missing values
    df_clean = df_clean[df_clean.isnull().sum(axis=1) <= 3]
    
    features = list(azdias_subset_1.columns)
    feat_info_clean = feat_info[feat_info['attribute'].isin(features)]
    # re-encode binary feature
    replace_values = {'W': 0, 'O': 1}
    df_clean = df_clean.replace({'OST_WEST_KZ':replace_values})
    
    _, multi_level_feature = feature_encoder(feat_info_clean) 
    
    df_clean = get_drop(df_clean, multi_level_feature)
    
    # Clena sampiles
    df_clean = azdias_data_clean_1(df_clean)
    
    df_clean = azdias_data_clean_2(df_clean)

    df_clean = mixed_feature_encoder(df_clean)
    
    # Return the cleaned dataframe.
    return df_clean

    


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[24]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.
np.float64
np.float
np.int
np.bool_
np.bool
float
bool 
int

fill_missing = Imputer(strategy='most_frequent')
azdias_clean_imputed = pd.DataFrame(fill_missing.fit_transform(azdias_clean))

azdias_clean_imputed.columns = azdias_clean.columns
azdias_clean_imputed.index = azdias_clean.index


# In[25]:


# Apply feature scaling to the general population demographics data.
scaler = StandardScaler() 
azdias_clean_scaled = scaler.fit_transform(azdias_clean_imputed)

azdias_clean_scaled = pd.DataFrame(azdias_clean_scaled, columns=list(azdias_clean_imputed))

azdias_clean_scaled.head()


# ### Discussion 2.1: Apply Feature Scaling
# 
# (Double-click this cell and replace this text with your own text, reporting your decisions regarding feature scaling.)
# I replaced all missing values with the most frequent value in that column using the Imputer method. All features are scaled using the StandardScaler.

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[26]:


# Apply PCA to the data.

pca = PCA()
pca.fit(azdias_clean_scaled)


# In[27]:


# Investigate the variance accounted for by each principal component.
plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.title("Variance explained")
plt.xlabel("Principal")
plt.ylabel("Ratio of variance")
plt.show()

plt.plot(range(len(pca.explained_variance_ratio_)),np.cumsum(pca.explained_variance_ratio_), '-')
plt.title("Cumulative Variance Explained")
plt.xlabel("Number")
plt.ylabel("Ratio of variance")
plt.show()


# In[28]:


# Re-apply PCA to the data while selecting for number of components to retain.

pca_30 = PCA(n_components=30)
azdias_pca = pca_30.fit_transform(azdias_clean_scaled)


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding dimensionality reduction. How many principal components / transformed features are you retaining for the next step of the analysis?)
# 
# I choose 30 principal components to analysis because it cuts features down by half while retaining a proper results

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[29]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.
def pca_weight(pca, weight_index): 
    df = pd.DataFrame(pca.components_, columns=list(azdias_clean_scaled.columns))
    weights = df.iloc[weight_index].sort_values(ascending=False)
    
    print('\n\n\nWeight:\t{0}\n\n**********************************************************************************************************\n'.format(weight_index))
    print(weights)
    return weights
   



pca_weight(pca, 0)


# In[30]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.
pca_weight(pca, 1)


# In[31]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.

pca_weight(pca, 2)


# In[32]:


pca_weight(pca, 3)


# ### Discussion 2.3: Interpret Principal Components
# 
# (Double-click this cell and replace this text with your own text, reporting your observations from detailed investigation of the first few principal components generated. Can we interpret positive and negative values from them in a meaningful way?)
# 
# PLZ8_ANTG3 and PLZ8_ANTG4 have a positive correlation. The family homes and the share of 6-10 family homes for macro-cells tend to increase

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[33]:


# Over a number of different cluster counts...
import _thread
import asyncio
import multiprocessing as mp
float
bool 
int
np.float64
np.float
np.int
np.bool_
np.bool
scores = []

ks = list(range(1,20))



def k_mean_score_threaded(data, n_cluster):
    print('Thread starting: Thread-{0}'.format(n_cluster))
    k_mean_score_result = k_mean_score(data, n_cluster)
    print('Thread done: Thread-{0}'.format(n_cluster))
    scores.append(k_mean_score_result)
    print("Score for Thread (Score, Thread) == {0}, {1}".format(k_mean_score_result, n_cluster))
    return k_mean_score_result

def k_mean_score(data, n_cluster):
    kmeans = KMeans(n_clusters = n_cluster)
    model = kmeans.fit(data)
    score = np.abs(model.score(data))
    return score
    # run k-means clustering on the data and...
    

     

thead_the_needle = []

for k in ks:
    print("Added item {0}".format(k))
#     thead_the_needle.append()
    k_mean_score_threaded(azdias_pca, k)
# loop = asyncio.get_event_loop()  
# loop.run_until_complete(asyncio.gather(*thead_the_needle))
    
    
scores


# In[34]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.
plt.plot(ks, scores, linestyle='-', marker='o')
plt.xlabel('K')
plt.ylabel('SSE')


# In[35]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
kmeans = KMeans(n_clusters = 14)
model = kmeans.fit(azdias_pca)
azdias_pred = model.predict(azdias_pca)


# ### Discussion 3.1: Apply Clustering to General Population
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding clustering. Into how many clusters have you decided to segment the population?)
# 
# it looks like K = 14
# on the plot

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[36]:


# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', sep=';')


# In[37]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.

customers_cleaned = clean_data(customers)

customers_cleaned_imputed = pd.DataFrame(fill_missing.transform(customers_cleaned))

customers_scaled = pd.DataFrame(scaler.transform(customers_cleaned_imputed),
                                            columns = list(customers_cleaned_imputed))

pca_customers = pca_30.transform(customers_scaled)

customers_pred = model.predict(pca_customers)


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[38]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.

figure, axs = plt.subplots(nrows=1, ncols=2, figsize = (10,5))
figure.subplots_adjust(hspace = 1, wspace=.3)

sns.countplot(customers_pred, ax=axs[0])
axs[0].set_title('Customer Clusters')
sns.countplot(azdias_pred, ax=axs[1])
axs[1].set_title('General Clusters')


# In[39]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?

cluster_1 = scaler.inverse_transform(pca_30.inverse_transform(model.cluster_centers_[4]))
over_cluster = pd.Series(data = cluster_1, index = customers_cleaned.columns)
over_cluster


# In[40]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?
cluster_2 = scaler.inverse_transform(pca_30.inverse_transform(model.cluster_centers_[13]))

down_under_cluster = pd.Series(data = cluster_2, index = customers_cleaned.columns)
down_under_cluster


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# (Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)
# 
# Cluster 1 is overrepresented in the customers data compared to general population data. Some characteristics of the group of population that are relative popular with the mail-order company:
# 
# Cluster 2 is underrepresented in the customers data. Some characteristics of the segment of the population that are relatively unpopular with the company:

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.
