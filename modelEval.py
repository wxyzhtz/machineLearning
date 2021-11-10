import numpy as np
import torch
import pandas as pd
from tools import toolkits
from mlxtend.plotting import heatmap

from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.inspection import plot_partial_dependence
# from pdpbox import pdp, get_dataset, info_plots
import graphviz
from sklearn.model_selection import cross_val_score

def dataLoad():
    pass


def machinelearningWorkflow(backTestPeriods, backtestStartDays=120, backtestEndDays=1):
    filePath = 'database\\fctValue'
    percent_select = [0.3, 0.3]

    for tradingDate in backTestPeriods:
        print(tradingDate + ' start.')
        trainingStart = toolkits().getXTradedate(tradingDate, backtestStartDays)
        trainingEnd = toolkits().getXTradedate(tradingDate, backtestEndDays)

        date_in_sample = toolkits().getTradingPeriod(trainingStart, trainingEnd)
        date_test = toolkits().getTradingPeriod(tradingDate, tradingDate)
        sampleDataParams = {
                    'filePath': filePath,
                    'date_in_sample': date_in_sample,
                    'percent_select': percent_select,
                    'dataList':['status', '1dayFutureRet', '1dayFutureRetTWAP', 'AuctionVolFactor',
                         'Beta', 'BookToPrice', 'CloseAuctionFactor_neu_Skew', 'EarningYield',
                         'Growth', 'Leverage', 'Liquidity', 'Momentum', 'net_buy_pct_PeriodsRet_ewma_2H', 'NonLinearSize',
                         'ResidualVolatility', 'RetSkewDaily_0950_rolling', 'Size', 'TopAmtFactor_1', 'UID', 'VolumeFactor_500'],
                    'targetlabel': '1dayFutureRet'
                }
        testDataParams = {
                    'filePath': filePath,
                    'date_in_test': date_test,
                    'dataList': ['status', '1dayFutureRet', '1dayFutureRetTWAP', 'AuctionVolFactor',
                                 'Beta', 'BookToPrice', 'CloseAuctionFactor_neu_Skew', 'EarningYield',
                                 'Growth', 'Leverage', 'Liquidity', 'Momentum', 'net_buy_pct_PeriodsRet_ewma_2H',
                                 'NonLinearSize',
                                 'ResidualVolatility', 'RetSkewDaily_0950_rolling', 'Size', 'TopAmtFactor_1', 'UID',
                                 'VolumeFactor_500'],
        }
        sampleData = toolkits().getSampleData(**sampleDataParams)
        testData = toolkits().getTestData(**testDataParams)
        train = sampleData.loc[:, 'AuctionVolFactor':'return_bin']
        test = testData.loc[:, 'AuctionVolFactor':'VolumeFactor_500']
        # for colName in sampleData.columns.tolist():
        #     scatterPlot(sampleData, colName)

        # corrPlot(train)

        print(train.shape, test.shape)
        target = 'return_bin'
        trainPredictors = [x for x in train.columns if x not in [target]]
        testPredictors = [x for x in test.columns if x not in [target]]

        # xgb1 = XGBClassifier(n_estimators=50, subsample=0.95, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')

        param_test1 = {
            'max_depth': list(range(3, 10, 2)),
            'min_child_weight': list(range(1, 6, 2))
        }
        gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=250, max_depth=5,
                                                        min_child_weight=1, gamma=0, subsample=0.9,
                                                        colsample_bytree=0.8,
                                                        objective='reg:logistic', nthread=4, scale_pos_weight=1,
                                                        seed=27, use_label_encoder=False),
                                param_grid=param_test1, scoring='roc_auc', n_jobs=4, cv=5)
        gsearch1.fit(train[trainPredictors], train[target])
        print(gsearch1.cv_results_['mean_test_score'], gsearch1.best_params_, gsearch1.best_score_)
        # trainingPred = xgb1.predict(train[trainPredictors])

        # trainingPredProba = xgb1.predict_proba(train[trainPredictors])[:, 1]

        # modelfitImportance(xgb1, features=trainPredictors, rawLabel=train[target], dtrain_predictions=trainingPred, dtrain_predprob=trainingPredProba)


        # print(xgb1.feature_importances_)
        # preds = xgb1.predict_proba(test[testPredictors])[:, 1]

        # print(sampleData, testData, )

def modelfitImportance(alg, features, rawLabel, dtrain_predictions, dtrain_predprob):

    # eval
    print("\n关于现在这个模型")
    print("准确率 : %.4g" % metrics.accuracy_score(rawLabel.values, dtrain_predictions))
    print("AUC 得分 (训练集): %f" % metrics.roc_auc_score(rawLabel.values, dtrain_predprob))

    # ft-imp
    feat_imp = pd.Series(alg.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(16, 5))
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

def missingPlot(df):
    fig, ax = plt.subplots(figsize=(15, 5))
    df_na = (df.isnull().sum() / len(df))
    df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[: 10]
    ax.bar(range(df_na.size), df_na, width=0.5)
    plt.xticks(range(df_na.size), df_na.index, rotation=0)
    plt.ylim([0, 1])
    plt.title('Top ten features with the most missing values')
    plt.ylabel('Missing ratio')
    plt.show()

def corrPlot(df):
    df = df.iloc[:, :-1]
    cm = np.corrcoef(df.values.T)
    hm = heatmap(cm, row_names=list(df.columns), column_names=list(df.columns), figsize=(20, 20))
    plt.title('Correlations Between the Different Features of the Data', fontsize=20)
    plt.show()

def scatterPlot(df, colName):
    plt.scatter(df[colName], df['1dayFutureRet'])
    plt.xlabel(colName)
    plt.ylabel('1dayFutureRet')
    plt.show()

if __name__ == '__main__':
    machinelearningWorkflow(['2021-08-20'])
