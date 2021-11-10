import numpy as np
import pandas as pd
from tools import toolkits
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import svm
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from database.others import fctNames
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import plot_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import torch.nn.functional as F
import torch


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=433, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=8)
        self.output = nn.Linear(in_features=8, out_features=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x

class ANNC(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.fc1 = nn.Linear(in_features=feature_size, out_features=60)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=60, out_features=10)
        self.drop2 = nn.Dropout(0.2)
        self.output = nn.Linear(in_features=10, out_features=2)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        x = self.output(x)
        # x = self.sigmoid(x)
        dout = F.softmax(x, dim=1)
        return dout

def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        # train
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['return_bin'].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        alg.fit(dtrain[predictors], dtrain['return_bin'], eval_metric='auc')

    # pred
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # eval
    print("\n关于现在这个模型")
    print("准确率 : %.4g" % metrics.accuracy_score(dtrain['return_bin'].values, dtrain_predictions))
    print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(dtrain['return_bin'], dtrain_predprob))

    # ft-imp
    feat_imp = pd.Series(alg.feature_importances_, index=list(dtrain[predictors].columns)).sort_values(ascending=False)
    plt.figure(figsize=(16, 5))
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

def label_data(data, percent_select=[0.5, 0.5]):
    data['return_bin']=np.NaN
    data=data.sort_values(by='5dayFutureRet', ascending=False)
    n_stock_select = np.multiply(percent_select, data.shape[0])
    n_stock_select = np.round(n_stock_select).astype(int)
    data.iloc[0:n_stock_select[0], -1]=1
    data.iloc[-n_stock_select[0]:, -1]=0
    # data.dropna(axis=0)
    return data

def label_data_10Group(data, keyword):
    data['return_bin']=round(data[keyword].rank(pct=True)*10)
    return data

# Press the green button in the gutter to run the script.

def machineLearningWorkflow(filePath, date_in_sample, date_test, percent_select, percent_cv, seed, method, svm_kernel, svm_c):
    data_in_sample = []
    for date in date_in_sample:
        fileName = filePath + '\\' + date + '.csv'
        dataDaily = pd.read_csv(fileName, index_col=0)
        print(dataDaily.columns.tolist())
        dataDaily = dataDaily.loc[:,
                    ['status', '5dayFutureRet', '5dayFutureRetTWAP',
                     'RetSkewDaily_0950_rolling', 'net_buy_pct_PeriodsRet_ewma_2H']]
        dataDaily = dataDaily.loc[dataDaily.status != 0, :]
        dataDaily = label_data(dataDaily, percent_select)
        dataDaily.dropna(inplace=True)
        data_in_sample.append(dataDaily)

    data_in_sampleDF = pd.concat(data_in_sample)

    X_in_sample = data_in_sampleDF.loc[:, 'RetSkewDaily_0950_rolling':'net_buy_pct_PeriodsRet_ewma_2H']
    print('The training size is {}'.format(X_in_sample.shape))

    y_in_sample = data_in_sampleDF.loc[:, 'return_bin']
    X_train, X_cv, y_train, y_cv = train_test_split(X_in_sample, y_in_sample, test_size=percent_cv,
                                                    random_state=seed)
    pca = decomposition.PCA(n_components=0.95)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_cv = pca.transform(X_cv)

    if method == 'SVM':
        model = svm.SVC(kernel=svm_kernel, C=svm_c)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_score_train = model.decision_function(X_train)
        y_pred_cv = model.predict(X_cv)
        y_score_cv = model.decision_function(X_cv)
        print('training set, accuracy=%.2f' % metrics.accuracy_score(y_train, y_pred_train))
        print('training set, AUC=%.2f' % metrics.roc_auc_score(y_train, y_score_train))
        print('cv set, accuracy=%.2f' % metrics.accuracy_score(y_cv, y_pred_cv))
        print('cv set, AUC=%.2f' % metrics.roc_auc_score(y_cv, y_score_cv))

    if method == 'RandomForestClassifier':
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_score_train = model.decision_function(X_train)
        y_pred_cv = model.predict(X_cv)
        y_score_cv = model.decision_function(X_cv)
        print('training set, accuracy=%.2f' % metrics.accuracy_score(y_train, y_pred_train))
        print('training set, AUC=%.2f' % metrics.roc_auc_score(y_train, y_score_train))
        print('cv set, accuracy=%.2f' % metrics.accuracy_score(y_cv, y_pred_cv))
        print('cv set, AUC=%.2f' % metrics.roc_auc_score(y_cv, y_score_cv))

    trueRetBag = []
    for date in date_test:
        print(date)
        fileName = filePath + '\\' + date + '.csv'
        dataDaily = pd.read_csv(fileName, index_col=0)
        dataDaily = dataDaily.loc[:,
                    ['status', '5dayFutureRet', '5dayFutureRetTWAP',
                     'RetSkewDaily_0950_rolling', 'net_buy_pct_PeriodsRet_ewma_2H']]
        dataDaily = dataDaily.loc[dataDaily.status != 0, :]
        # dataDaily.dropna(inplace=True)
        X_current_day = dataDaily.loc[:, 'RetSkewDaily_0950_rolling':'net_buy_pct_PeriodsRet_ewma_2H']
        X_current_day = pca.transform(X_current_day)
        trueRet = pd.DataFrame(dataDaily.loc[:, '5dayFutureRetTWAP'])

        if method == 'SVM':
            y_pred_curr_day = model.predict(X_current_day)
            y_score_curr_day = model.decision_function(X_current_day)

        trueRet['pred'] = y_pred_curr_day
        trueRet['pred_prob'] = y_score_curr_day
        trueRet['date'] = [date] * len(trueRet)
        print(trueRet)
        trueRetBag.append(trueRet)

    fullTrueRet = pd.concat(trueRetBag)
    fullTrueRet.to_csv('results\\SVMTestResults_' + date + '.csv')
    pos = fullTrueRet.loc[fullTrueRet.pred == 1, :].groupby('date').mean()
    neg = fullTrueRet.loc[fullTrueRet.pred == 0, :].groupby('date').mean()

def getLabel(rawDF):
    rawDF['return_bin'] = np.NaN
    upStocks = rawDF.ret[rawDF.ret>0]
    downStocks = rawDF.ret[rawDF.ret<0]

    commonLength = min(len(upStocks), len(downStocks))
    upStocks = upStocks[upStocks.rank(ascending=False)<=commonLength]
    downStocks = downStocks[downStocks.rank()<=commonLength]

    topUpStocks = upStocks[upStocks.rank(pct=True)>0.7]
    topDownStocks = downStocks[downStocks.rank(pct=True)<0.3]
    rawDF.loc[topUpStocks.index.tolist(), 'return_bin'] = 1
    rawDF.loc[topDownStocks.index.tolist(), 'return_bin'] = 0
    rawDF.dropna(inplace=True)

    return rawDF


def getSampleData(filePath, date_in_sample, targetDF, targetDates, factorList, benchmark):
    data_in_sample = []
    for date in date_in_sample:
        fileName = filePath + '\\' + date + '.csv'
        dataDaily = pd.read_csv(fileName, index_col=0)
        dataDaily = dataDaily.loc[:,
                    ['status'] + factorList]
        dataDaily = dataDaily.loc[dataDaily.status != 0, :]
        dataDaily.replace(np.inf, np.NaN, inplace=True)
        dataDaily.fillna(dataDaily.mean(), inplace=True)
        ret = targetDF.loc[targetDates[date], :]/targetDF.loc[date, :] - 1
        benchmark_ret = benchmark.loc[targetDates[date], 'closePrice']/benchmark.loc[date, 'closePrice'] - 1
        dataDaily['ret'] = ret.loc[list(map(lambda x:x[:6], dataDaily.index.tolist()))].tolist() - benchmark_ret
        dataDaily.dropna(inplace=True)
        dataDaily = getLabel(dataDaily)
        data_in_sample.append(dataDaily)
    data_in_sampleDF = pd.concat(data_in_sample)
    data_in_sampleDF.dropna(inplace=True)
    return data_in_sampleDF

def getSampleDataR(filePath, date_in_sample, targetDF, targetDates, factorList, benchmark):
    data_in_sample = []
    for date in date_in_sample:
        fileName = filePath + '\\' + date + '.csv'
        dataDaily = pd.read_csv(fileName, index_col=0)
        dataDaily = dataDaily.loc[:,
                    ['status'] + factorList]
        dataDaily = dataDaily.loc[dataDaily.status != 0, :]
        dataDaily.replace(np.inf, np.NaN, inplace=True)
        dataDaily.fillna(dataDaily.mean(), inplace=True)
        ret = targetDF.loc[targetDates[date], :]/targetDF.loc[date, :] - 1
        benchmark_ret = benchmark.loc[targetDates[date], 'closePrice']/benchmark.loc[date, 'closePrice'] - 1
        dataDaily['ret'] = ret.loc[list(map(lambda x:x[:6], dataDaily.index.tolist()))].tolist() - benchmark_ret
        dataDaily.dropna(inplace=True)
        # dataDaily = getLabel(dataDaily)
        dataDaily['return_bin'] = dataDaily['ret'].tolist()
        data_in_sample.append(dataDaily)
    data_in_sampleDF = pd.concat(data_in_sample)
    data_in_sampleDF.dropna(inplace=True)
    return data_in_sampleDF

def getSampleData10(filePath, date_in_sample, targetDF, targetDates, factorList, benchmark):
    data_in_sample = []
    for date in date_in_sample:
        fileName = filePath + '\\' + date + '.csv'
        dataDaily = pd.read_csv(fileName, index_col=0)
        dataDaily = dataDaily.loc[:,
                    ['status'] + factorList]
        dataDaily = dataDaily.loc[dataDaily.status != 0, :]
        dataDaily.replace(np.inf, np.NaN, inplace=True)
        dataDaily.fillna(dataDaily.mean(), inplace=True)
        ret = targetDF.loc[targetDates[date], :]/targetDF.loc[date, :] - 1
        benchmark_ret = benchmark.loc[targetDates[date], 'closePrice']/benchmark.loc[date, 'closePrice'] - 1
        dataDaily['ret'] = ret.loc[list(map(lambda x:x[:6], dataDaily.index.tolist()))].tolist() - benchmark_ret
        dataDaily.dropna(inplace=True)
        dataDaily['return_bin'] = np.round(dataDaily['ret'].rank(pct=True)*10-0.0000000001).astype(int)
        #　dataDaily = getLabel(dataDaily)
        data_in_sample.append(dataDaily)
    data_in_sampleDF = pd.concat(data_in_sample)
    data_in_sampleDF.dropna(inplace=True)
    return data_in_sampleDF

def getSampleDataLinear(filePath, date_in_sample, targetDF, targetDates, factorList, benchmark):
    data_in_sample = []
    for date in date_in_sample:
        fileName = filePath + '\\' + date + '.csv'
        dataDaily = pd.read_csv(fileName, index_col=0)
        dataDaily = dataDaily.loc[:,
                    ['status'] + factorList]
        dataDaily = dataDaily.loc[dataDaily.status != 0, :]
        dataDaily.replace(np.inf, np.NaN, inplace=True)
        dataDaily.fillna(dataDaily.mean(), inplace=True)
        ret = targetDF.loc[targetDates[date], :]/targetDF.loc[date, :] - 1
        benchmark_ret = benchmark.loc[targetDates[date], 'closePrice']/benchmark.loc[date, 'closePrice'] - 1
        dataDaily['ret'] = ret.loc[list(map(lambda x:x[:6], dataDaily.index.tolist()))].tolist() - benchmark_ret
        dataDaily.dropna(inplace=True)
        # dataDaily = getLabel(dataDaily)
        data_in_sample.append(dataDaily)
    data_in_sampleDF = pd.concat(data_in_sample)
    data_in_sampleDF.dropna(inplace=True)
    return data_in_sampleDF

def getTestData(filePath, date_in_test, factorList):
    fileName = filePath + '\\' + date_in_test[0] + '.csv'
    dataDaily = pd.read_csv(fileName, index_col=0)
    dataDaily = dataDaily.loc[:,
                ['status']+factorList]
    dataDaily = dataDaily.loc[dataDaily.status != 0, :]
    dataDaily.replace(np.inf, np.NaN, inplace=True)
    dataDaily.loc[dataDaily.count(axis=1) < 2] = np.NaN
    dataDaily.dropna(how='all', inplace=True)
    dataDaily.fillna(dataDaily.mean(), inplace=True)
    dataDaily.dropna(inplace=True)
    return dataDaily

def xgboostWorkflow(data_in_sampleDF, data_in_testDF, tradingDate, factorList):
    print(tradingDate)
    train = data_in_sampleDF.loc[:, factorList[0]:'return_bin']
    test = data_in_testDF.loc[:, factorList[0]:'return_bin']
    print(train.shape, test.shape)
    target = 'return_bin'
    trainPredictors = [x for x in train.columns if x not in [target]]
    testPredictors = [x for x in test.columns if x not in [target]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[target], test_size=0.15,
                                                    random_state=42)

    # scores = cross_val_score(xgb1, data_in_sampleDF[trainPredictors], data_in_sampleDF[target], cv=5)
    # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    # data_in_testDF.to_csv('results\\xgb\\xgboost_' + tradingDate + '.csv')
    # modelfit(xgb1, train, test, predictors)

    # param_test1 = {
    #     'max_depth': range(3, 10, 2),
    #     'min_child_weight': range(1, 6, 2)}
    #
    # gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=50, seed=10, eval_metric='mlogloss', use_label_encoder=False),
    #                         param_grid=param_test1, scoring='roc_auc', cv=5)
    # gsearch1.fit(train[predictors], train[target])
    # print(gsearch1.cv_results_['mean_test_score'], gsearch1.best_params_, gsearch1.best_score_)
    parameters = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'max_depth': [8, 10, 12, 15],
        'gamma': [0.001, 0.005, 0.01, 0.02],
        'random_state': [42]
    }
    eval_set = [(X_train, y_train), (X_cv, y_cv)]
    model = xgb.XGBClassifier(eval_set=eval_set, objective='reg:squarederror', verbose=False)
    clf = GridSearchCV(model, parameters)

    clf.fit(X_train, y_train)

    print(f'Best params: {clf.best_params_}')
    print(f'Best validation score = {clf.best_score_}')

    # xgb1 = XGBClassifier(n_estimators=200, max_depth=3, min_child_weight=1, subsample=0.95, use_label_encoder=False, eval_metric='mlogloss')
    # xgb1.fit(data_in_sampleDF[trainPredictors], data_in_sampleDF[target])
    # preds = xgb1.predict_proba(data_in_testDF[testPredictors])[:, 1]
    # data_in_testDF['pred_prob'] = preds
    # data_in_testDF['tradingDate'] = [tradingDate]*len(preds)
    # data_in_testDF = data_in_testDF.loc[:, ['tradingDate', '5dayFutureRet', '5dayFutureRetTWAP', 'pred_prob']]


def xgboostRegrWorkflow(data_in_sampleDF, data_in_testDF, factorList):
    train = data_in_sampleDF.loc[:, factorList + ['target']]
    test = data_in_testDF.loc[:, factorList]
    target = 'target'
    trainPredictors = [x for x in train.columns if x not in [target]]
    testPredictors = [x for x in test.columns if x not in [target]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[target], test_size=0.15,
                                                    random_state=42)

    other_params = {'verbosity': 0,
             'objective': 'reg:pseudohubererror',
             'eval_metric': 'mae',
             'subsample': 0.9,
             'colsample_bytree': 0.8,
             'tree_method': 'gpu_hist',
             'eta': 0.1,
             'max_depth': 10,
             'gamma': 0,
             'min_child_weight': 1}

    xgbRegr = xgb.XGBRegressor(**other_params)
    # optimized_GBM = GridSearchCV(estimator=xgbRegr, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    xgbRegr.fit(X_train, y_train)
    preds=xgbRegr.predict(test.loc[:, testPredictors])
    # xgbRegr.fit(X_train, y_train)
    y_cv_pred = xgbRegr.predict(X_cv)
    mse = mean_squared_error(y_cv_pred, y_cv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_cv_pred, y_cv)
    print("RMSE : % f" % (rmse))
    print('mse: {:.3}'.format(mse))
    print('mae: {:.3}'.format(mae))
    print('r2 {:.3}'.format(r2_score(y_cv_pred, y_cv)))

    # # scores = cross_val_score(xgbRegr, X_train, y_train, cv=5)
    # # print('Accuracy: {:.3f} ± {:.3f}'.format(np.mean(scores), 2 * np.std(scores)))
    # preds = xgbRegr.predict(X_test)

    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # param = {'verbosity': 0,
    #          'objective': 'reg:pseudohubererror',
    #          'eval_metric': 'mae',
    #          'subsample': 0.8,
    #          'colsample_bytree': 0.8,
    #          'tree_method': 'gpu_hist',
    #          'eta': 0.1,
    #          'max_depth': 5,
    #          'gamma': 0,
    #          'min_child_weight': 1}
    #
    # bst = xgb.cv(param, dtrain, nfold=3, num_boost_round=1000, early_stopping_rounds=50)
    # fig = plt.figure(figsize=(12, 4))
    # fig.suptitle('The Mean Absolute Error (MAE) of the training data and validation data')
    # plt.subplot(121)
    # plt.plot(bst['train-mae-mean'], label='train')
    # plt.plot(bst['test-mae-mean'], label='validation')
    # plt.xlabel('Runs')
    # plt.ylabel('MAE')
    # plt.legend()
    # plt.subplot(122)
    # plt.plot(bst['train-mae-mean'], label='train')
    # plt.plot(bst['test-mae-mean'], label='validation')
    # plt.yscale('log')
    # plt.xlabel('Runs')
    # plt.ylabel('MAE in log scale')
    # plt.legend()
    # plt.subplots_adjust(wspace=0.3)
    # plt.show()
    return pd.Series(preds, index=test.index.tolist())

def linearRegrWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    linearRegr = LinearRegression(fit_intercept=False)
    linearRegr.fit(X_train, y_train)
    preds = linearRegr.predict(test.loc[:, trainPredictors])
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries

def logRegrWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    Regr = LogisticRegression(fit_intercept=False, max_iter=1000)
    Regr.fit(X_train, y_train)
    predsTrain = Regr.predict(X_train)
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = Regr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    preds = Regr.predict_proba(test.loc[:, factorList])[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def SVMWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    Regr = SVC(kernel=kernel, probability=True)
    Regr.fit(X_train, y_train)
    predsTrain = Regr.predict(X_train)
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = Regr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    preds = Regr.predict_proba(test.loc[:, factorList])[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def DTCWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    Regr = tree.DecisionTreeClassifier()
    Regr.fit(X_train, y_train)
    predsTrain = Regr.predict(X_train)
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = Regr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    preds = Regr.predict_proba(test.loc[:, factorList])[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def XGBCWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    Regr = xgb.XGBClassifier(use_label_encoder=False, max_depth=3, min_child_weight=1, subsample=0.95, objective='binary:logistic')
    Regr.fit(X_train, y_train)
    predsTrain = Regr.predict(X_train)
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = Regr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    # preds = Regr.predict_proba(test.loc[:, factorList])
    # preds = (preds.dot(np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]))).reshape(1, -1).tolist()[0]
    preds = Regr.predict_proba(test.loc[:, factorList])[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def XGBRWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    Regr = xgb.XGBRegressor(n_estimators=200, use_label_encoder=False, max_depth=3, min_child_weight=1, subsample=0.95, )
    Regr.fit(X_train, y_train)
    predsTrain = Regr.predict(X_train)
    Training_accuracy = 'not need'
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = Regr.predict(X_cv)
    accuracy = 'not need'
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    # preds = Regr.predict_proba(test.loc[:, factorList])
    # preds = (preds.dot(np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]))).reshape(1, -1).tolist()[0]
    preds = Regr.predict(test.loc[:, factorList])#[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def XGBCMultiWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    Regr = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, max_depth=5, min_child_weight=1, subsample=0.95,)
    Regr.fit(X_train, y_train)
    predsTrain = Regr.predict(X_train)
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = Regr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    preds = Regr.predict_proba(test.loc[:, factorList])
    preds = (preds.dot(np.array([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]))).reshape(1, -1).tolist()[0]
    # preds = Regr.predict_proba(test.loc[:, factorList])[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def RFWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    Regr = RandomForestClassifier(n_estimators=500, max_features=20, min_samples_split=50, min_samples_leaf=10)
    Regr.fit(X_train, y_train)

    predsTrain = Regr.predict(X_train)
    Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))

    predsCV = Regr.predict(X_cv)
    accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation Accuracy: {} MAE: {}'.format(accuracy, mae))
    preds = Regr.predict_proba(test.loc[:, factorList])[:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, accuracy, mae

def linearRegrWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors], data_in_sampleDF[targetName],
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)
    Regr = LinearRegression(fit_intercept=False)
    Regr.fit(X_train, y_train)

    predsTrain = Regr.predict(X_train)
    # Training_accuracy = np.sum([i == j for i, j in zip(predsTrain, y_train)])/len(y_train)
    Training_mae = mean_absolute_error(predsTrain, y_train)
    print('Training MAE: {}'.format(Training_mae))

    predsCV = Regr.predict(X_cv)
    # accuracy = np.sum([i == j for i, j in zip(predsCV, y_cv)])/len(y_cv)
    mae = mean_absolute_error(predsCV, y_cv)
    print('Cross-validation MAE: {}'.format(mae))
    preds = Regr.predict(test.loc[:, factorList]) # [:, 1]
    predSeries = pd.Series(preds, test.index.tolist())
    return predSeries, Training_mae, mae

def evaluate_accuracy(x,y,net):
    with torch.no_grad():
        out = net(x)
        correct= (out.argmax(1) == y).sum().item()
        n = y.shape[0]
    return correct / n

def evaluate_mae(x,y,net):
    with torch.no_grad():
        out = net(x)
        mae = mean_squared_error(out.argmax(1).detach().numpy(), y.detach().numpy())
    return mae

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def ANNWorkflow(data_in_sampleDF, data_in_testDF, factorList, targetName, kernel='rbf'):
    epochs = 200

    train = data_in_sampleDF.loc[:, factorList + [targetName]]
    test = data_in_testDF.loc[:, factorList]
    trainPredictors = [x for x in train.columns if x not in [targetName]]

    X_train, X_cv, y_train, y_cv = train_test_split(data_in_sampleDF[trainPredictors].values, data_in_sampleDF[targetName].values,
                                                    test_size=0.15,
                                                    random_state=42)
    print(X_train.shape)

    X_train = torch.FloatTensor(X_train)
    X_cv = torch.FloatTensor(X_cv)
    y_train = torch.LongTensor(y_train)
    y_cv = torch.LongTensor(y_cv)
    X_test = torch.FloatTensor(test.values)

    model = ANNC(feature_size=len(factorList))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_arr = []
    best_acc = 0
    for i in range(epochs):
        y_hat = model.forward(X_train)
        loss = criterion(y_hat, y_train)
        loss_arr.append(loss)
        acc = evaluate_accuracy(X_train, y_train, model)
        acc_cv = evaluate_accuracy(X_cv, y_cv, model)
        if i % 20 == 0:

            print(f'Epoch: {i} Loss: {loss}')
            print('training acc: {}, cv acc: {}'.format(acc, acc_cv))

        if best_acc < acc_cv:
            best_acc = acc_cv
            # print("save model, the cv_acc is {}%".format(accTmp))
            torch.save(model.state_dict(), 'model.pth')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    with torch.no_grad():
        model = ANNC()
        model.load_state_dict(torch.load('model.pth'))
        y_test_pred = model(X_test)
        print("The selected model cv_acc is {}%".format(best_acc*100))
        Training_accuracy = evaluate_accuracy(X_train, y_train, model)
        Training_mae = evaluate_mae(X_train, y_train, model)
        cv_accuracy = evaluate_accuracy(X_cv, y_cv, model)
        cv_mae = evaluate_mae(X_cv, y_cv, model)
        print('Training Accuracy: {} MAE: {}'.format(Training_accuracy, Training_mae))
        print('Cross-validation Accuracy: {} MAE: {}'.format(cv_accuracy, cv_mae))
    predSeries = pd.Series(y_test_pred[:, 1].detach().numpy(), test.index.tolist())
    return predSeries, Training_accuracy, Training_mae, cv_accuracy, cv_mae

def getResRet(X, y):
    regr = LinearRegression(fit_intercept=False)
    regr.fit(X, y)
    predParams = regr.coef_
    resRet = y - regr.predict(X)
    return resRet, predParams

def getSampleDataV1(filePath, date_in_sample, targetDF, targetDates, factorList):
    data_in_sample = []
    paramsdict = {}
    for date in date_in_sample:
        fileName = filePath + '\\' + date + '.csv'
        dataDaily = pd.read_csv(fileName, index_col=0)
        dataDaily = dataDaily.loc[:,
                    ['status'] + factorList]
        dataDaily = dataDaily.loc[dataDaily.status == 0, :]
        dataDaily.replace(np.inf, np.NaN, inplace=True)
        dataDaily.fillna(dataDaily.mean(), inplace=True)
        ret = targetDF.loc[targetDates[date], :]/targetDF.loc[date, :] - 1
        dataDaily['ret'] = ret.loc[list(map(lambda x:x[:6], dataDaily.index.tolist()))].tolist()
        dataDaily.dropna(inplace=True)
        dataDaily['ResRet'], params = getResRet(dataDaily[factorList], dataDaily['ret'])
        dataDaily['ResRet'] = np.round(dataDaily['ResRet'].rank(pct=True)*10).astype(int)
        data_in_sample.append(dataDaily)
        paramsdict[date] = params
    data_in_sampleDF = pd.concat(data_in_sample)
    data_in_sampleDF.dropna(inplace=True)
    estParams = pd.DataFrame(paramsdict).T.mean()
    return data_in_sampleDF, estParams

def getTestDataV1(filePath, date_in_test, factorList):
    fileName = filePath + '\\' + date_in_test[0] + '.csv'
    dataDaily = pd.read_csv(fileName, index_col=0)
    dataDaily = dataDaily.loc[:,
                ['status']+factorList]
    dataDaily = dataDaily.loc[dataDaily.status != 0, :]
    dataDaily.replace(np.inf, np.NaN, inplace=True)
    dataDaily.loc[dataDaily.count(axis=1) < 2] = np.NaN
    dataDaily.dropna(how='all', inplace=True)
    dataDaily.fillna(dataDaily.mean(), inplace=True)
    dataDaily.dropna(inplace=True)
    return dataDaily


if __name__ == '__main__':
    fctType = '技术指标'
    fctInfo = pd.read_excel('F:\\research\\多因子\\机器学习\\结果统计.xlsx', sheet_name='筛选后优矿因子')
    factorList = fctInfo.loc[fctInfo['大类']==fctType, '具体因子'].tolist()
    closePrice=toolkits().readRawData('closePrice', 'database\\tradingData')
    benchmark = pd.read_csv('database\\tradingData\\SH000905.csv', index_col=0)
    tradingDates = toolkits().getTradingPeriod('2016-01-01', '2021-08-20')
    method = 'SVM'
    filePath = 'database\\fctValue'
    percent_select = [0.3, 0.3]
    percent_cv = 0.1
    target = closePrice
    # factorList = ['CloseAuctionFactor', 'Davis_Momentum', 'net_buy_pct_PeriodsRet_ewma_2H', 'OpenAuctionVolFactor', 'Reversal']
    # factor = pd.read_csv('database\\fctValue\\2018-01-18.csv', index_col=0)
    # factorList = factor.columns[6:].tolist()
    factorDict = {}
    factorLinearDict = {}
    factorNonLinearDict = {}
    stat = {}
    tradingDates = pd.Series(toolkits().cutTimeSeries(tradingDates, freq='week'), index=toolkits().cutTimeSeries(tradingDates, freq='week'))
    for tradingDate in tradingDates[50:]:
        print('{} starts.'.format(tradingDate))
        trainingDates = tradingDates[:tradingDate].iloc[-51:-1]
        targetDates = tradingDates.shift(-1)[:tradingDate].iloc[-51:-1]
        testDate = tradingDate
        # for tradinDate in trainingDates:

        sampleData= getSampleData(filePath, trainingDates, target, targetDates, factorList, benchmark)
        testData = getTestDataV1(filePath, [tradingDate], factorList)
        logisticPred, Training_accuracy, Training_mae, CV_accuracy, CV_mae = ANNWorkflow(sampleData, testData, factorList, 'return_bin')
        # logisticPred, Training_mae, CV_mae = linearRegrWorkflow(sampleData, testData, factorList, 'return_bin')

        factorDict[tradingDate] = logisticPred
        stat[tradingDate] = {'Training_accuracy':Training_accuracy, 'Training_mae':Training_mae, 'CV_accuracy': CV_accuracy, 'CV_mae':CV_mae}
        # stat[tradingDate] = {'Training_mae':Training_mae, 'CV_mae':CV_mae}

    fctDF = pd.DataFrame(factorDict).T
    fctDF.to_csv('results\\fct\\' + fctType + '_ANNC_Pred50_blog.csv')

    statDF = pd.DataFrame(stat).T
    statDF.to_csv('results\\stat\\' + fctType + '_ANNC_Pred50_blog.csv')

    # fctDFL = pd.DataFrame(factorLinearDict).T
    # fctDFL.to_csv('results\\fct\\linear.csv')

    # fctDFNL = pd.DataFrame(factorNonLinearDict).T
    # fctDFNL.to_csv('results\\fct\\nonLinearXGB.csv')
    # print(fctDF, fctDFL, fctDFNL)

