# Databricks notebook source
# MAGIC %pip install catboost
# MAGIC %pip install catboost xgboost
# MAGIC import pandas as pd
# MAGIC import matplotlib.pyplot as plt
# MAGIC import seaborn as sns
# MAGIC import numpy as np
# MAGIC from sklearn.preprocessing import MinMaxScaler
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC from sklearn.linear_model import LinearRegression
# MAGIC from sklearn.metrics import mean_squared_error, r2_score
# MAGIC from sklearn.preprocessing import OneHotEncoder
# MAGIC from sklearn.compose import ColumnTransformer
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC from sklearn.tree import DecisionTreeRegressor
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC from sklearn.pipeline import make_pipeline
# MAGIC from sklearn.svm import SVR
# MAGIC from sklearn.ensemble import RandomForestRegressor
# MAGIC from sklearn.ensemble import GradientBoostingRegressor
# MAGIC from catboost import CatBoostRegressor
# MAGIC from xgboost import XGBRegressor
# MAGIC import warnings

# COMMAND ----------

df = spark.read.csv('s3://bigdata-carsales-2015/', header=True)
df = df.toPandas()
df.head()

# COMMAND ----------

df.shape

# COMMAND ----------

df.head()

# COMMAND ----------

df.isna().sum()


# COMMAND ----------

df.dropna(inplace=True)

# COMMAND ----------

df.isna().sum()

# COMMAND ----------

df['saledate'][0]

# COMMAND ----------



# COMMAND ----------

df.dtypes

# COMMAND ----------

df.columns

# COMMAND ----------

def infer_and_set_types(dataframe):
    for column in dataframe.columns:
        first_val = dataframe[column].iloc[0]
        try:
            pd.to_numeric(dataframe[column], errors='raise')
            dataframe[column] = dataframe[column].astype(int)
        except ValueError:
                dataframe[column] = dataframe[column].astype(str)
                print(f"{column} is treated as object (string or mixed type).")

    return dataframe


# COMMAND ----------

df=infer_and_set_types(df)
#df['year'] = pd.to_datetime(df['year'].astype(str), errors='coerce')
df['saledate'] = df['saledate'].str.rsplit(' ', 1).str[0]
df['saledate'] = pd.to_datetime(df['saledate'], format='%a %b %d %Y %H:%M:%S GMT%z', errors='coerce')


# COMMAND ----------

df['saledate'] = pd.to_datetime(df['saledate'], utc=True)
df['sale_Date'] = df['saledate'].dt.date
df['sale_year'] = df['saledate'].dt.year
df['sale_Weekday'] = df['saledate'].dt.day_name()
df['sale_Time'] = df['saledate'].dt.strftime('%H:%M:%S')
df['Month'] = df['saledate'].dt.month_name() 

# COMMAND ----------

df.isna().sum()

# COMMAND ----------

df.head()

# COMMAND ----------

df['make'].value_counts()

# COMMAND ----------

df_toyo=df[df['make']=='Toyota']
df_to_s=df_toyo[df_toyo['model'].isin(['Camry', 'Corolla'])]

# COMMAND ----------

#most of my friends prefer to buy the toyota.(Used cars)

# COMMAND ----------

df.head()

# COMMAND ----------

plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(data=df, x='odometer', y='sellingprice')
plt.title('Relationship Between Odometer Reading, Price, and State by Vehicle Type')
plt.xlabel('Odometer Reading')
plt.ylabel('Price')
#plt.legend(title='Vehicle/State', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# COMMAND ----------

corr_matrix = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix of DataFrame')
plt.show()

# COMMAND ----------

avg_price_by_year = df.groupby('year')['sellingprice'].mean().reset_index()
#avg_price_by_year['year'] = avg_price_by_year['year'].dt.year
plt.figure(figsize=(12, 8))
bar_chart = sns.barplot(data=avg_price_by_year, x='year', y='sellingprice', palette='deep')
plt.title('Average Selling Price by Year')
plt.xlabel('Year')
plt.ylabel('Average Selling Price ($)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# COMMAND ----------



# COMMAND ----------

avg_price_by_maker = df.groupby('make')['sellingprice'].mean().reset_index()
avg_price_by_maker = avg_price_by_maker.sort_values(by='sellingprice', ascending=False)
avg_10=avg_price_by_maker.head(10)
plt.figure(figsize=(12, 8)) 
bar_chart = sns.barplot(data=avg_10, x='make', y='sellingprice', palette='deep')
plt.title('Average Selling Price by maker(Top_10)')
plt.xlabel('Year')
plt.ylabel('Average Selling Price ($)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# COMMAND ----------



# COMMAND ----------


count_by_maker = df.groupby('make')['sellingprice'].count().reset_index()
count_by_maker.columns=['make','count']
count_by_maker = count_by_maker.sort_values(by='count', ascending=False)
count_10=count_by_maker.head(10)
plt.figure(figsize=(12, 8)) 
bar_chart = sns.barplot(data=count_10, x='make', y='count', palette='deep')
plt.title('Total Number by maker(Top_10)')
plt.xlabel('Year')
plt.ylabel('Count of vechical')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# COMMAND ----------

df.columns

# COMMAND ----------

count_by_month

# COMMAND ----------

plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(data=df, x='sellingprice', y='condition')
plt.title('Price vs Condition')
plt.xlabel('Price')
plt.ylabel('Condition')

# Show the plot
plt.show()

# COMMAND ----------

plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(data=df, x='sellingprice', y='mmr')
plt.title('Price vs mmr')
plt.xlabel('Price')
plt.ylabel('Condition')

# Show the plot
plt.show()

# COMMAND ----------

df_to_s.shape

# COMMAND ----------

df_to_s.columns

# COMMAND ----------

df

# COMMAND ----------

y = car_prices['sellingprice']
X = car_prices[['year', 'make', 'model', 'odometer', 'condition','mmr','state']]

categorical_features = ['make', 'model','state']
numerical_features = ['year', 'odometer', 'condition','mmr']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# COMMAND ----------



Linear_Re = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Linear_Re.fit(X_train, y_train)

y_pred = Linear_Re.predict(X_test)
print(f'Linear Regression R² score: {r2_score(y_test, y_pred)}')

# COMMAND ----------

from sklearn.model_selection import cross_val_score
scores_LR = cross_val_score(Linear_Re, X, y, cv=5, scoring='r2')
print("R² scores for each fold:", scores_LR)

# Plotting the R² scores
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(scores_LR) + 1), scores_LR, marker='o', linestyle='-', color='b')
plt.title('Cross-Validation R² Scores')
plt.xlabel('Fold Number')
plt.ylabel('R² Score')
plt.grid(True)
plt.xticks(range(1, len(scores_LR) + 1))
plt.show()

# COMMAND ----------


tree_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_pipeline.fit(X_train, y_train)

y_pred = tree_pipeline.predict(X_test)
print(f'Decision Tree R² score: {r2_score(y_test, y_pred)}')

# COMMAND ----------

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_pipeline, X, y, cv=5, scoring='r2')
print("R² scores for each fold:", scores)

# Plotting the R² scores
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='b')
plt.title('Cross-Validation R² Scores')
plt.xlabel('Fold Number')
plt.ylabel('R² Score')
plt.grid(True)
plt.xticks(range(1, len(scores) + 1))
plt.show()

# COMMAND ----------

df

# COMMAND ----------

def predict_car_price(model):
    inputs = {}
    for feature in numerical_features:
        inputs[feature] = float(input(f"Enter {feature}: "))
    for feature in categorical_features:
        inputs[feature] = input(f"Enter {feature}: ")
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)
    return prediction[0]

# COMMAND ----------

predicted_price = predict_car_price(Linear_Re)
print(f"The predicted selling price of the car is: ${predicted_price:.2f}")

# COMMAND ----------

df[df['model']=='Camry']

# COMMAND ----------

####the price of the index : 562 of the Camry based on the conditionS ACTUAL PRICE : $13500 PREDICTED WAS : $13246.73 only

# COMMAND ----------

#lets create the models for individual maker or for particular some specific models So our model can predict .close to the actual value. And also by increasing the features for better accuracy . 
#Lets create the model only for the toyota Sience most of my friends are purchasing the second hand Toyota(Cord)

# COMMAND ----------

df_Toy = df[(df['make'] == 'Toyota') & (df['model'].isin(['Camry', 'Corolla']))]

# COMMAND ----------

df_Toy

# COMMAND ----------

y = df_Toy['sellingprice']
X = df_Toy[['year', 'make', 'model', 'odometer', 'condition','mmr','state','trim','body','transmission','interior','Month']]

categorical_features = ['make', 'model','state','trim','body','transmission','interior','Month']
numerical_features = ['year', 'odometer', 'condition','mmr']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
linear_Toy = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_Toy.fit(X_train, y_train)

y_pred = linear_Toy.predict(X_test)
print(f'Linear Regression R² score: {r2_score(y_test, y_pred)}')
scores_LR = cross_val_score(linear_Toy, X, y, cv=5, scoring='r2')
print("R² scores for each fold:", scores_LR)
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(scores_LR) + 1), scores_LR, marker='o', linestyle='-', color='b')
plt.title('Cross-Validation R² Scores')
plt.xlabel('Fold Number')
plt.ylabel('R² Score')
plt.grid(True)
plt.xticks(range(1, len(scores_LR) + 1))
plt.show()

# COMMAND ----------

predicted_price = predict_car_price(linear_Toy)
print(f"The predicted selling price of the car for Toyota is: ${predicted_price:.2f}")

# COMMAND ----------

print("The Actual selling price at index 557 is:", df_Toy.iloc[557]['sellingprice'])
df_Toy.iloc[557]

# COMMAND ----------



