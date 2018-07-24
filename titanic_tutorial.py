import pandas as pd

test=pd.read_csv("C:/Users/TRIPS1/Documents/Python/titanic/titanictest.csv")
train=pd.read_csv("C:/Users/TRIPS1/Documents/Python/titanic/titanictrain.csv")

#Print the dimensions of both datasets
print("Dimensions of train: {}".format(train.shape))
print("Dimensions of test: {}".format(test.shape))

train.head()


import matplotlib.pyplot as plt

#Pivot and view survived by sex
sex_pivot=train.pivot_table(index="Sex", values="Survived")
sex_pivot.plot.bar()
plt.show()

#Pivot and view survived by class
class_pivot=train.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar()
plt.show()

#Explore Age
train["Age"].describe()
survived=train[train["Survived"]==1]
died=train[train["Survived"]==0]
survived["Age"].plot.hist(alpha=0.5,color='green',bins=50)
died["Age"].plot.hist(alpha=0.5,color='red',bins=50)
plt.legend(['Survived','Died'])
plt.show()

#Preprocessing on Age
def process_age(df,cut_poijts,label_names):
    df["Age"]=df["Age"].fillna(-0.5)
    df["Age_categories"]=pd.cut(df["Age"],cut_points,labels=label_names)
    return df
cut_points=[-1,0,5,12,18,35,60,100]
label_names=["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train=process_age(train,cut_points,label_names)
test=process_age(test,cut_points,label_names)

pivot=train.pivot_table(index="Age_categories",values='Survived')
pivot.plot.bar()
plt.show()

#train["Pclass"].value_counts()

#Create Dummy variables for all classifiers to make them boolean
def create_dummies(df,column_name):
    dummies=pd.get_dummies(df[column_name],prefix=column_name)
    df=pd.concat([df,dummies],axis=1)
    return df

for column in ["Pclass","Sex","Age_categories"]:
    train=create_dummies(train,column)
    test=create_dummies(test,column)


#Create the first machine learning model. Logistic Regression
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']
lr.fit(train[columns],train["Survived"])


#Testing the first machine learning model
holdout = test # from now on we will refer to this
               # dataframe as the holdout data

from sklearn.model_selection import train_test_split

all_X = train[columns]
all_y = train['Survived']

train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.20,random_state=0)

#Training the model on the new train set
lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)


#Check accuracy of model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y, predictions)
#accuracy score 81%

#Doing 10 fold cross validation
from sklearn.model_selection import cross_val_score

lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv=10)
scores.sort()
accuracy = scores.mean()

print(scores)
print(accuracy)


#Create a prediction model for holdout
lr = LogisticRegression()
lr.fit(all_X,all_y)
holdout_predictions = lr.predict(holdout[columns])

#Creating a Kaggle output file
holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv("C:/Users/TRIPS1/Documents/Python/titanic/submission.csv",index=False)


