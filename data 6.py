import pandas as pd
df=pd.read_csv('C:/Users/admin/Downloads/archive (4)/tested.csv')
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt



df=df.drop('PassengerId',axis=1)
df=df.drop('Cabin',axis=1)
df=df.drop('Name',axis=1)
df=df.drop('Ticket',axis=1)
# df=df.drop('Fare',axis=1)
from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['Fare'])
plt.show()

df['Fare'].fillna(df['Fare'].mean(),inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['Sex'])
df['Sex']=le.transform(df['Sex'])
le = LabelEncoder()
le.fit(df['Embarked'])
df['Embarked']=le.transform(df['Embarked'])
df['Age'].fillna(df['Age'].median(),inplace=True)
print(df.describe())
df1=df.drop('Survived',axis=1)
et = ExtraTreesClassifier()
et.fit(df1,df['Survived'])
print(et.feature_importances_)

feat_imp=pd.Series(et.feature_importances_,index=df1.columns)
feat_imp.nlargest(7).plot(kind='barh')
plt.show()
df=df.drop('Age',axis=1)
df=df.drop('Embarked',axis=1)
df=df.drop('Pclass',axis=1)

x=df.drop('Survived',axis=1)
y=df['Survived']
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2,k='all')
bestfeatures.fit(x,y)
dfscore=pd.DataFrame(bestfeatures.scores_)
dfcolumns=pd.DataFrame(x.columns)
featuresScores=pd.concat([dfcolumns,dfscore],axis=1)
featuresScores.columns=['Features','Score']
print(featuresScores)


# for col in df.columns.values:
#     sns.boxplot(df[col])
#     plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
df['SibSp']=pd.cut(df['SibSp'],2,labels=[0,1])


dt=DecisionTreeClassifier(random_state=0)
rf=RandomForestClassifier(random_state=0)
gb=GradientBoostingClassifier(n_estimators=10)
lr=LogisticRegression(random_state=0)
sc=svm.SVC(random_state=0)
gnb=GaussianNB()
mnb=MultinomialNB()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, train_size=0.3)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(accuracy_score(y_test,y_pred))

rf.fit(x_train,y_train)
y_pred1=rf.predict(x_test)
print(accuracy_score(y_test,y_pred1))

gb.fit(x_train,y_train)
y_pred2=gb.predict(x_test)
print(accuracy_score(y_test,y_pred2))

lr.fit(x_train,y_train)
y_pred3=lr.predict(x_test)
print(accuracy_score(y_test,y_pred3))

sc.fit(x_train,y_train)
y_pred4=sc.predict(x_test)
print(accuracy_score(y_test,y_pred4))

gnb.fit(x_train,y_train)
y_pred5=gnb.predict(x_test)
print(accuracy_score(y_test,y_pred5))

mnb.fit(x_train,y_train)
y_pred=mnb.predict(x_test)
print(accuracy_score(y_test,y_pred))

from sklearn.naive_bayes import BernoulliNB
bn=BernoulliNB()
bn.fit(x_train,y_train)
pred=bn.predict(x_test)
print(accuracy_score(y_test,pred))

'''        Survived      Pclass         Sex  ...       Parch        Fare    Embarked
count  418.000000  418.000000  418.000000  ...  418.000000  418.000000  418.000000
mean     0.363636    2.265550    0.636364  ...    0.392344   35.627188    1.401914
std      0.481622    0.841838    0.481622  ...    0.981429   55.840500    0.854496
min      0.000000    1.000000    0.000000  ...    0.000000    0.000000    0.000000
25%      0.000000    1.000000    0.000000  ...    0.000000    7.895800    1.000000
50%      0.000000    3.000000    1.000000  ...    0.000000   14.454200    2.000000
75%      1.000000    3.000000    1.000000  ...    0.000000   31.500000    2.000000
max      1.000000    3.000000    1.000000  ...    9.000000  512.329200    2.000000

[8 rows x 8 columns]
[0.00765891 0.9235773  0.01423188 0.01281303 0.01736183 0.01566261
 0.00869444]
'''