import pandas as pd
import numpy as np
# s1 = pd.Series([1,2,3,4])
# print(s1)
# s2 = pd.Series(np.arange(5), index=['a', 'b', 'c', 'd', 'e'])
# print(s2)
# s3 = pd.Series({"name": "zhangsan", "age": 27, "tel": 10086})
# print(s3)
# s4 = pd.Series(1., index=list("abcde"))
# print(s4)
# s2 = pd.Series([1,2,3,4])
# result = s2[1:] + s2[:-1]
# print(result)
# print(result.isna())
# print('..........')
# print(result.fillna(1.0,limit=1))
# print('...')
# print(result.dropna())   # inplace = True 则返回到result进行替换，不会有返回值


# P_Pclass1_Survived,P_Pclass2_Survived,P_Pclass3_Survived = 1,1,1   #    3档
# P_SexM_Survived,P_SexF_Survived = 1,1      #    2档
# P_AgeC_Survived,P_AgeA_Survived,P_AgeE_Survived = 1,1,1      #    标准： 儿童：0-18  成年：19-50 老人：61-
# P_EmbarkedS_Survived,P_EmbarkedC_Survived,P_EmbarkedQ_Survived = 1,1,1
# P_Sibsp_Survived = 1    #    兄弟姐妹
# P_Parch_Survived = 1    #    父母孩子
from pandas import read_csv
filename = 'D:\PycharmProjects\data/titanic/train.csv'
# names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
data = read_csv(filename)
dataset = data.fillna(-999)
sum = 0
for i in range(0,dataset.shape[0]):
    sum = sum + dataset['Survived'][i]
P_Survived_num = sum
P_Survived = P_Survived_num / dataset.shape[0]    # 342/891 = 0.3838383838....
sum1,sum2,sum3,sum4,sum5,sum6= 0,0,0,0,0,0
for i in range(0,dataset.shape[0]):
    if dataset['Pclass'][i] == 1:
        sum1 += 1
        if dataset['Survived'][i] == 1:
            sum4 += 1
    elif dataset['Pclass'][i] == 2:
        sum2 += 1
        if dataset['Survived'][i] == 1:
            sum5 += 1
    else:
        sum3 += 1
        if dataset['Survived'][i] == 1:
            sum6 += 1
P_Pclass_Survived = [sum4/sum1,sum5/sum2,sum6/sum3]
# [0.6296296296296297, 0.47282608695652173, 0.24236252545824846]
sum1,sum2,sum3,sum4,sum5,sum6= 0,0,0,0,0,0
for i in range(0,dataset.shape[0]):
    if dataset['Sex'][i] == 'male':
        sum1 += 1
        if dataset['Survived'][i] == 1:
            sum4 += 1
    elif dataset['Sex'][i] == 'female':
        sum2 += 1
        if dataset['Survived'][i] == 1:
            sum5 += 1
P_Sex_Survived = {'male':sum4/sum1,'female':sum5/sum2}
# 0.18890814558058924 0.7420382165605095
# print(dataset)
sum1,sum2,sum3,sum4,sum5,sum6= 0,0,0,0,0,0
sum7,sum8 = 0,0
for i in range(0,dataset.shape[0]):
    if dataset['Age'][i] <= 17 and dataset['Age'][i] > 0 :
        sum1 += 1
        if dataset['Survived'][i] == 1:
            sum4 += 1
    elif dataset['Age'][i] >= 18 and dataset['Age'][i] <= 50 :
        sum2 += 1
        if dataset['Survived'][i] == 1:
            sum5 += 1
    elif dataset['Age'][i] >= 51:
        sum3 += 1
        if dataset['Survived'][i] == 1:
            sum6 += 1
    else:
        sum7 += 1
        if dataset['Survived'][i] == 1:
            sum8 += 1
P_Age_Survived = [sum4/sum1,sum5/sum2,sum6/sum3,sum8/sum7]
# [0.5398230088495575, 0.3854748603351955, 0.34375, 0.2937853107344633]
sum1,sum2,sum3,sum4,sum5,sum6= 0,0,0,0,0,0
for i in range(0,dataset.shape[0]):
    if dataset['Embarked'][i] == 'S':
        sum1 += 1
        if dataset['Survived'][i] == 1:
            sum4 += 1
    elif dataset['Embarked'][i] == 'C':
        sum2 += 1
        if dataset['Survived'][i] == 1:
            sum5 += 1
    else:
        sum3 += 1
        if dataset['Survived'][i] == 1:
            sum6 += 1
P_Embarked_Survived = {'S':sum4/sum1,'C':sum5/sum2,'Q':sum6/sum3,-999:1}
# {'S': 0.33695652173913043, 'C': 0.5535714285714286, 'Q': 0.4050632911392405}
TP,FP,TN,FN = 0,0,0,0
for i in range(dataset.shape[0]):
    if dataset['Age'][i] <= 17 and dataset['Age'][i] > 0 :
        PAg = P_Age_Survived[0]
    elif dataset['Age'][i] >= 18 and dataset['Age'][i] <= 50 :
        PAg = P_Age_Survived[1]
    elif dataset['Age'][i] >= 51:
        PAg = P_Age_Survived[2]
    else:
        PAg = P_Age_Survived[3]
    if P_Embarked_Survived[dataset['Embarked'][i]] == -999:
        P_Embarked_Survived[dataset['Embarked'][i]] = 1
    PS = P_Survived * P_Pclass_Survived[dataset['Pclass'][i]-1] * P_Sex_Survived[dataset['Sex'][i]]  * PAg #* P_Embarked_Survived[dataset['Embarked'][i]]
    PD = (1-P_Survived) * (1-P_Pclass_Survived[dataset['Pclass'][i]-1]) * (1-P_Sex_Survived[dataset['Sex'][i]]) * (1-PAg)  #* (1-P_Embarked_Survived[dataset['Embarked'][i]])
    P_SurvivedG = (PS > PD)
    if P_SurvivedG == dataset['Survived'][i]:
        if P_SurvivedG == 1:
            TP += 1
        else:
            TN += 1
    else:
        if P_SurvivedG == 1:
            FP += 1
        else:
            FN += 1
ACU = (TP+TN)/(FP+FN+TP+TN)
print(TP,TN,FP,FN)
print(ACU)

print('.....')
filename = 'D:\PycharmProjects\data/titanic/test.csv'
# names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
data = read_csv(filename)
dataset = data.fillna(-999)
datafin = read_csv('D:\PycharmProjects\data/titanic/gender_submission.csv')
for i in range(dataset.shape[0]):
    if dataset['Age'][i] <= 17 and dataset['Age'][i] > 0 :
        PAg = P_Age_Survived[0]
    elif dataset['Age'][i] >= 18 and dataset['Age'][i] <= 50 :
        PAg = P_Age_Survived[1]
    elif dataset['Age'][i] >= 51:
        PAg = P_Age_Survived[2]
    else:
        PAg = P_Age_Survived[3]
    if P_Embarked_Survived[dataset['Embarked'][i]] == -999:
        P_Embarked_Survived[dataset['Embarked'][i]] = 1
    PS = P_Survived * P_Pclass_Survived[dataset['Pclass'][i]-1] * P_Sex_Survived[dataset['Sex'][i]]  * PAg #* P_Embarked_Survived[dataset['Embarked'][i]]
    PD = (1-P_Survived) * (1-P_Pclass_Survived[dataset['Pclass'][i]-1]) * (1-P_Sex_Survived[dataset['Sex'][i]]) * (1-PAg) # * (1-P_Embarked_Survived[dataset['Embarked'][i]])
    P_SurvivedG = int((PS > PD))
    datafin['PassengerId'][i] = i+1
    datafin['Survived'][i] = P_SurvivedG
    # print(P_SurvivedG,end=' ')
datafin.to_csv('D:\PycharmProjects\data/titanic/gender_submission1.csv')