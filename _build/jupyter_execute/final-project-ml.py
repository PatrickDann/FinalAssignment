#!/usr/bin/env python
# coding: utf-8

# # Predicting Low Fat Cheese
# #### by Patrick Dann
# 

# ## Introduction
# 
# Back to [Intro](intro.md).

# In this Project I will predict the fat content of cheese based on it's properties. This is a classification problem since we will be grouping the cheese into a category such as; low fat, high fat, etc. 
# 
# Predicting the fat content of cheese is desiarable since the fat content may be important for cooking and how the cheese properties will influsence a dish. Cheese ia also a source of saturated fat which would be link to the total fat content of the cheese and is seen as undisiarable by many people. Additionally the fat content will greatly effect the calorie content of the cheese. Therefore, when manufacturing a cheese the fat content should be considered and knowing what properties contribute to making a low fat cheese is valuable. 
# 
# We will be looking for cheese with a low fat content and see what properties are predictive for such cheeses. 

# ## Exploratory Data Analysis

# In[1]:


# Import libraries needed 

import altair as alt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC, SVR
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# I will Select the cheese data and split it into test and train sets so the golden rule is not vialated.

# In[2]:


# Read in the cheese data
cheese_df=pd.read_csv(r"C:\Users\ichir\Documents\final-assignment\cheese_data.csv")
cheese_df


# In[3]:


# Now we will make the training and test sets 
train_df, test_df = train_test_split(cheese_df, test_size=0.2, random_state=77)
train_df.head()


# I will seperat the features form the target and select the ones required in the analysis. 
# The target `y` column will be `FatLevel` and I will use `MoisturePercent`, `Organic`, `MilkTypeEn`, `MilkTreatmentTypeEn` and `RindTypeEn` as the `x` since these are relevant to the manufaturing process and cheese properties. 

# In[4]:


# Create the train and test splits

X_train= train_df.drop(columns=['CheeseId', 'ManufacturerProvCode', 'FlavourEn', 'CharacteristicsEn', 'CheeseName', 'FatLevel'])
y_train= train_df['FatLevel']
y_train= y_train.map({'lower fat': 1, 'higher fat': 0}).astype(int)

X_test= test_df.drop(columns=['CheeseId', 'ManufacturerProvCode', 'FlavourEn', 'CharacteristicsEn', 'CheeseName', 'FatLevel' ])
y_test= test_df['FatLevel']
y_test= y_test.map({'lower fat': 1, 'higher fat': 0}).astype(int)

X_train


# Now I will look at the features and see dtypes for the variables.

# In[5]:


# take a look at the dtypes of the train set 
X_train.info()


# We can see that there are 729 entries and the `MoisturePercent`, `CategoryTypeEn`, `MilkTypeEn`, `MilkTreatmentTypeEn`, and `RindTypeEn` all have null value. 
# 
# Aditionally we can see that there are 5 categorical features, 2 numerical.

# Now I will have a deaper look at the features.

# In[6]:


# looking at the prdinal column 
X_train['CategoryTypeEn'].unique()


# In[7]:


# describe the features 
X_train.describe


# Now I'll look at the numerical columns statistics

# In[8]:


# describe the numerical features 
X_train.describe()


# Lets look at the `Organic` column since it apears to be binary.

# In[9]:


X_train['Organic'].unique()


# The `Organic` column does appear to be binary and not numerical. That leaves us with one numberical column, `MoisturePercent`

# There are null values in the columns `MoisturePercent`, `CategoryTypeEn`, `MilkTypeEn`, `MilkTreatmentTypeEn`, and `RindTypeEn` so these will have to be imputed.

# I want to visualize some of the categorical features and see what kind of distributions I am dealing with.

# In[10]:


# Visualize MilkTypeEn column distribution 
MilkType_plot = alt.Chart(cheese_df).mark_bar().encode(
                    alt.X('MilkTypeEn', title="Milk Type", sort='y'),
                    alt.Y('count()', title='Number of counts', stack=None),
                    alt.Color('FatLevel', title='Fat Level')).properties(title="Milk Type Distribution"
                                                                        ).facet('FatLevel')

MilkType_plot


# In[11]:


# Visualize MilkTreatmentTypeEn column distribution 
MilkTreatmentType_plot = alt.Chart(cheese_df).mark_bar().encode(
                    alt.X('MilkTreatmentTypeEn', title="Milk Treatment Type", sort='y'),
                    alt.Y('count()', title='Number of counts', stack=None),
                    alt.Color('FatLevel', title='Fat Level')).properties(title="Milk Treatment Type Distribution"
                                                                        ).facet('FatLevel')

MilkTreatmentType_plot


# In[12]:


# Visualize ManufacturingTypeEn column distribution 
ManufacturingType_plot = alt.Chart(cheese_df).mark_bar().encode(
                    alt.X('ManufacturingTypeEn', title="Manufacturing Type", sort='y'),
                    alt.Y('count()', title='Number of counts', stack=None),
                    alt.Color('FatLevel', title='Fat Level')).properties(title="Manufacturing Type Distribution"
                                                                        ).facet('FatLevel')

ManufacturingType_plot


# Now lets see if there is any relationship between Moisture Percent and Fat level

# In[13]:


# Plot MoisturePercent againt FatLevel 
MoisturePercent_plot = alt.Chart(cheese_df).mark_boxplot().encode(
                        alt.X('MoisturePercent', title='Moisture Percent'),
                        alt.Y('FatLevel', title='Fat Level')).properties(title='Moisture Percent Relationship with Fat Level')
MoisturePercent_plot


# It seems that cheese with Lower Fat generally have a higher Moisture Percent. This makes sense since this would mean that the cheese has a higher amount of water by weight. 

# ## Methods and Results

# I will first make a `DummyClassifier` so I can use this as a baseline to compare the final model too. 

# In[14]:


# building and scoreing the DummyClassifier
dummy_model = DummyClassifier(strategy = 'prior')

scores = cross_validate(dummy_model, X_train, y_train, cv=5, return_train_score=True)

dummy_scores = pd.DataFrame(scores)

dummy_scores


# I want to see the mean score to use to compare to my model

# In[15]:


dummy_scores.mean()


# The mean test score is the same as the mean train score. This means that the model is probalby underfitting

# I have already identified the Numerical, Binary, and Categorical features in the `X_train` so now I will define them in lists to make the pipelines

# In[16]:


# Defining the numerical, binary, ordinal and categorical features 
numeric_feats = ['MoisturePercent' ]
binary_feats = ['Organic']
ordinal_feats = []
categorical_feats = ['ManufacturingTypeEn', 'MilkTreatmentTypeEn', 'MilkTypeEn','CategoryTypeEn', 'RindTypeEn'] 



# Now I will make the Transformers for the Column Transformers. Since there are missing values I will have to use `SimpleImputer` and I will have to use the `OneHotEncoder` for the binary and categorical features so they can be used in the model.

# In[17]:


# making the Numerical Transformer
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

# Making the Binary Transformer 
binary_transformer = Pipeline(
    steps=[("imputer",SimpleImputer(strategy='constant')),("onehot", OneHotEncoder(drop='if_binary', dtype=int))])

# Making the Categorical Transformer
categorical_transformer =Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent", fill_value="missing")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])


# Now I will make the Column Transformer with the transformers and their designated features 

# In[18]:


# Making the Columntransformer 
col_transformer = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_feats),
        ("cat", categorical_transformer, categorical_feats),
        ("binary", binary_transformer, binary_feats)])


# Now I will make the Pipeline, I will use the `LogisticRegression` since we are looking at a classificatioin problem

# In[19]:


# Making the Pipepline
lr_pipe = Pipeline(
    steps=[("preprocessor", col_transformer), ("reg", LogisticRegression(class_weight="balanced", max_iter=1000))])


# In[20]:


# Fitting the pipeline on the training set
lr_pipe.fit(X_train, y_train)


# In[21]:


# Finding the accuracy precision and recall of the model
                          
lr_scores = pd.DataFrame(cross_validate(lr_pipe, X_train, y_train, cv=5, return_train_score=True, scoring=['accuracy', 'precision', 'recall']))
lr_scores


# In[22]:


lr_scores.mean()


# Now I will make a `SVC` model and see how it compares to the `LogisticRegression` model 

# In[23]:


# Making the SVC model
SVC_pipe = Pipeline(
    steps=[("preprocessor", col_transformer), ("svc", SVC(class_weight="balanced"))])


# In[24]:


# Fitting the model on the training data
SVC_pipe.fit(X_train, y_train)


# In[25]:


# Scoring the SVC modal
SVC_scores = pd.DataFrame(cross_validate(SVC_pipe, X_train, y_train, cv=5, return_train_score=True, scoring=['accuracy', 'precision', 'recall']))
SVC_scores


# In[26]:


SVC_scores.mean()


# In[27]:


# comparing the 2 models 
print('SVC scores')
print(SVC_scores.mean())
print('LogisticRegresssion scores')
print(lr_scores.mean())  


# The `LogisticRegresssion` performs better than the `SVC` model on the accuracy and recall scores however the `SVC` model performs better on the precision score. 
# 
# In this case the recall is more important than the precision since it would be more detrimental if we mistakenly produce a high fat cheese (false negative) than miss a low fat cheese (false posititive)
# 
# I will tune the hyperparameters for the `LogisticRegression` estimator to find the best model but first I will have to find which parameters to tune. I will tune on recall since I iddentified that is the most important.
# 

# In[28]:


# searching the parameters for the LogisticRegression model
LogisticRegression().get_params().keys()


# In[29]:


param_grid =  {  
        'reg__C': [100, 10, 1.0, 0.1, 0.01],
        'reg__penalty': ['l2'],
        'reg__solver': ['newton-cg', 'lbfgs', 'liblinear']}

scoring={
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}


# Tuning the Hyperparameters C, penaly, and solver with GridSearchCV
grid_search = GridSearchCV(lr_pipe, param_grid, cv=5, n_jobs=-1, verbose=3, return_train_score=True, scoring=scoring, refit='recall_score')

# Fitting to the training set
grid_search.fit(X_train, y_train)


# In[30]:


grid_search.best_score_


# In[31]:


# Finiding the best parameters from the grid search 
grid_search.best_params_


# In[32]:


# finding the best model
best_model = grid_search.best_estimator_


# In[33]:


# Fitting the best model
best_model.fit(X_train, y_train)


# In[34]:


# scoring the accuracy of the best model on the test set
best_model.score(X_test, y_test)


# In[35]:


# veiwing the classification report
print(classification_report(y_test, best_model.predict(X_test)))


# In[36]:


# compared to the baseline `DummyClassifier
dummy_model.fit(X_train, y_train)

print(classification_report(y_test, dummy_model.predict(X_test)))


# After hyperparameter tuning the best model has an accuracy of 0.75, recall of 0.74. 
# 
# I want to see which features influence the prediciton the most so I have to find all the features and compare them to there particular coefficients. I will do this with `.coef_` 

# In[37]:


# reading in the best LogisticRegression model
lr_reg=LogisticRegression(C=100, class_weight='balanced', max_iter=1000, solver='newton-cg')

# Transforming the X_train with the column transformer from the pipeline
X_train_transformed = col_transformer.transform(X_train)

# fitting the best LogisticRegression model 
lr_reg.fit(X_train_transformed, y_train)

# veiwing the coefficients
lr_coeffs =lr_reg.coef_
lr_coeffs


# In[38]:


# Find the new categorical columns created by the column transformer
new_cols = col_transformer.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_feats)

# Make an object with all the features
columns = numeric_feats + list(new_cols) + binary_feats
columns


# In[39]:


lr_reg.intercept_


# In[40]:


# Predict on the test set
predicted_y = best_model.predict(X_test)

# find the Probability 
proba_y = best_model.predict_proba(X_test)

# View the probability of low fat of eeach cheese name in a dataframe
lr_probs = pd.DataFrame({
            "Cheese name":test_df['CheeseName'],
             "true y":y_test, 
             "pred y": predicted_y.tolist(),
             "prob_LowFat": proba_y[:, 1].tolist()})
lr_probs.sort_values(by='prob_LowFat', ascending=False)


# In[41]:


# Veiwing the coefficients of the features in a dataframe
data = {'features': columns, 'coefficients':lr_coeffs[0]}
pd.DataFrame(data).sort_values(by='coefficients', ascending=False)


# ## Discussion

# The best model has an accuracy of 0.75, recall of 0.74, f1 scores of 0.73 and percision of 0.73. This is better than the baseline `DummyClassifier` which had a accuracy of 0.66, precision of 0.33, recall score of 0.50, and f1 scores of 0.40. 
# 
# The main metric I used to score the model was the recall score since the recall score, or sensitivity, finds the True positive rate or in this case the rate at which we properly identify a cheese as Low fat. I decided that this metric is important since if we want to manufacure low fat cheese and use this model to predict the manufacturing properties requierd to make low fat cheese it would be more detrimental to mistakenly make a high fat cheese than miss a potential low fat cheese. 
# 
# Looking at the features and coefficients we can see that the `MilkTypeEn_Cow, Goat and Ewe` and `MoisturePercent` significantly contribute to the model in predicting low fat cheese based on the data. This makes sense to me since a higher moister percent would mean a higher weight by water and thus lower weight by fat. Also I'd expect milk type to have the greatest impact on fat content since the majority if not all the fat in the cheese would be comming from the milk. Additionaly, We can also see that the `MilkType_Ewe and Cow` feature contributes most negatively, or in other words significantly contributes to predicting High fat. Some features that minumily contributed to the prediciton of Low/High fat content were `MilkTypeEn_Buffalo Cow` and `ManufacturingTypeEn_Industrial`. The buffalo cow milk type not contributing to the model suprises me since I would think that Buffalo milk having different properties than cow milk and thus affecting the fat content. However, Industrial Manufacturing feature does not suprise me since I would expect Industiral processes to be quite addaptable and be able to produce many different kinds of High fat or Low fat cheeses. 
# 
# I am curious about how the `FlavourEn` and `Characteristics` features would have impacted the model. I think the using these features would have improved that model since I suspect that fat content, flavour and characteristics of cheese are closely tied together. 
# 
# Furthur questions about this dataset would be if you could predict `FlavourEN` of the cheese from the other features of the cheese. Thi would be interesting since prediction the tast/smell of a cheese would be very desirable> Then the model would be albe to be used to find features which are predictive of a particular taste profiles of cheese. 
# 

# ## References 

# {cite}'7'
# {cite}'8'
