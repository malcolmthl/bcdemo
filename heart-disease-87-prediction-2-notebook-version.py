# Laptop / Desktop
# cd C:\Users\malco\PycharmProjects\aiforcancers
# cd C:\Users\malco\PycharmProjects\aiforcancers\heart-failure-prediction

#VSCode
# cd C:\Users\malco\breastcancer_dashboard_2

#uvicorn heart-disease-87-prediction-2-notebook-version:app --reload
# http://127.0.0.1:8000
#http://127.0.0.1:8000/input_stats/33/'F'/'ASY'/100/100/0/'LVH'/100/'N'/2.5/'Down'/
from fastapi import FastAPI
from typing import Optional



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import random

# from plotly.subplots import make_subplots
# import plotly.express as px
# import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score

df = pd.read_csv('heart.csv')
df_orig = df.copy()
df

cate = []
numerical = []
for i in df.columns:
    if df[i].dtype == 'object':
        cate.append(i)
    else:
        numerical.append(i)

df_numerical = df[numerical]
df_categorical = df[cate]
# fig = plt.figure(figsize=(12, 9))
# sns.heatmap(df_numerical.corr(),annot=True)

# get unique vars for each column
colName = df.columns.to_list()
colNameX = colName[:-1]
randomList1 = []
randomList2 = []
randomList3 = []
count = 0
for i in range(df.shape[1]):  # shorten

    colNamei = colName[i]
    exec(f'colNamei_uniq{count} = list(np.unique(df.loc[:, colNamei]))')
    exec(f'colName{count} = colNamei')
    exec(f'temp_colNamei_uniq = colNamei_uniq{count}')
    print('colName{}: [{}], uniq: {}'.format(i, colNamei, temp_colNamei_uniq))

    print(temp_colNamei_uniq)
    #     colNamei_uniq_sampled1 = random.sample(colNamei_uniq,1)[0]
    #     colNamei_uniq_sampled2 = random.sample(colNamei_uniq,1)[0]
    #     colNamei_uniq_sampled3 = random.sample(colNamei_uniq,1)[0]

    #     randomList1.append(colNamei_uniq_sampled1)
    #     randomList2.append(colNamei_uniq_sampled2)
    #     randomList3.append(colNamei_uniq_sampled3)
    #     print('colNamei_uniq:',colNamei_uniq)
    count += 1
colName


useR_inputList_long = []

colName = df.columns.to_list()

app = FastAPI()
# http://127.0.0.1:8000/input_stats/33/'F'/'ASY'/100/100/0/'LVH'/100/'N'/2.5/'Down'/
# 33/'F'/'ASY'/100/100/0/'LVH'/100/'N'/2.5/'Down'/
@app.get("/input_stats/{Age}/{Sex}/{ChestPainType}/{RestingBP}/{Cholesterol}/{FastingBS}/{RestingECG}/{MaxHR}/{ExerciseAngina}/{Oldpeak}/{ST_Slope}/")
def get_userDataframe(*,
                      Age,
                      Sex,
                      ChestPainType,
                      RestingBP,
                      Cholesterol,
                      FastingBS,
                      RestingECG,
                      MaxHR,
                      ExerciseAngina,
                      Oldpeak,
                      ST_Slope):
    # mal get randomly generated stats
    all_stats = [Age, Sex,ChestPainType,RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]

    for stats in all_stats:
        useR_inputList_long.append(stats)

    df_user = pd.DataFrame([useR_inputList_long])
    df_user.columns = colNameX

    # encode train_df
    df_train = df.copy()

    lbl_encode = LabelEncoder()
    for i in df_categorical.columns:
        df_train[i] = df_train[[i]].apply(lbl_encode.fit_transform)
    # mal encode User
    for i in df_categorical.columns:
        df_user[i] = df_user[[i]].apply(lbl_encode.fit_transform)
    df_user

    # train test split
    df_train = df_train.drop(columns='HeartDisease')
    target = df['HeartDisease'].copy()

    x_train, x_test, y_train, y_test = train_test_split(df_train, target, test_size=0.15, random_state=42)

    # scale, make df
    scaled_features_train = StandardScaler().fit_transform(x_train.values)
    scaled_features_test = StandardScaler().fit_transform(x_test.values)
    x_train_scaled = pd.DataFrame(scaled_features_train, index=x_train.index, columns=df_train.columns)
    x_test_scaled = pd.DataFrame(scaled_features_test, index=x_test.index, columns=df_train.columns)

    # scale user input

    # df_user = df_user.drop(columns='HeartDisease')
    scaled_features_test_userInput = StandardScaler().fit_transform(df_user.values)
    x_test_scaled_userInput = pd.DataFrame(scaled_features_test_userInput, index=df_user.index,
                                           columns=df_train.columns)

    gnb = GaussianNB()
    cv = cross_val_score(gnb, x_train, y_train, cv=5)
    print('gnb cv:', cv)
    print('gnb cvmean:', cv.mean())

    lr = LogisticRegression(max_iter=2000)
    cv = cross_val_score(lr, x_train_scaled, y_train, cv=5)
    print('lr cv:', cv)
    print('lr cvmean:', cv.mean())

    dt = tree.DecisionTreeClassifier(random_state=1)
    cv = cross_val_score(dt, x_train_scaled, y_train, cv=5)
    print('dt cv:', cv)
    print('dt cvmean:', cv.mean())

    knn = KNeighborsClassifier(n_neighbors=5)
    cv = cross_val_score(knn, x_train_scaled, y_train, cv=5)
    print('knn cv:', cv)
    print('knn cvmean:', cv.mean())

    svc = SVC(probability=True)
    cv = cross_val_score(svc, x_train_scaled, y_train, cv=5)
    print('svc cv:', cv)
    print('svc cvmean:', cv.mean())

    def clf_performance(classifier, model_name):
        print(model_name)
        print('Best Score: ' + str(classifier.best_score_))
        print('Best Parameters: ' + str(classifier.best_params_))

    lr = LogisticRegression()
    param_grid = {'max_iter': [2000],
                  'penalty': ['l1', 'l2'],
                  'C': np.logspace(-4, 4, 20),
                  'solver': ['liblinear']}
    clf_lr = GridSearchCV(lr, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    best_clf_lr = clf_lr.fit(x_train_scaled, y_train)
    clf_performance(best_clf_lr, 'LogisticRegression')

    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': [3, 5, 7, 9],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                  'p': [1, 2]}
    clf_knn = GridSearchCV(knn, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    best_clf_knn = clf_knn.fit(x_train_scaled, y_train)
    clf_performance(best_clf_knn, 'KNN')

    svc = SVC(probability=True)
    param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1, .5, 1, 2, 5, 10],
                                      'C': [.1, 1, 10, 100, 1000]},
                                     {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
                                     {'kernel': ['poly'], 'degree': [2, 3, 4, 5], 'C': [.1, 1, 10, 100, 1000]}]
    clf_svc = GridSearchCV(svc, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    best_clf_svc = clf_svc.fit(x_train_scaled, y_train)
    clf_performance(best_clf_svc, 'SVC')

    rf = RandomForestClassifier(random_state=1)
    param_grid = {'n_estimators': [50, 100, 200],
                  'criterion': ['gini', 'entropy'],
                  'min_samples_leaf': [1, 2, 3],
                  }
    clf_rf = GridSearchCV(rf, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
    best_clf_rf = clf_rf.fit(x_train_scaled, y_train)
    clf_performance(best_clf_rf, 'RandomForestClassifier')

    knn = KNeighborsClassifier(algorithm='auto', n_neighbors=9, p=1, weights='uniform').fit(x_train_scaled, y_train)
    # pred_knn = knn.predict(x_test_scaled) # Orig
    pred_knn = list(knn.predict(x_test_scaled_userInput))  # User Prediction
    print("User Prediction (KNN): {}".format(pred_knn))

    rf = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_leaf=2).fit(x_train_scaled, y_train)
    # pred_rf = rf.predict(x_test_scaled) # Orig
    pred_rf = list(rf.predict(x_test_scaled_userInput))  # User Prediction
    print("User Prediction (rf): {}".format(pred_rf))
    # accuracy_score(y_test, pred_rf) # Orig to check accuracy score

    return 0






