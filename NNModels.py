import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression

class NeuralNetRegressionModel():
    def trainAndGetResults(X, Y, Y_baseline, baseline=False):
        results = {'clf': [] , 'x': [], 'X_train': [], 'Y_pred': [], 'Y_test': [], 'mse': [], 'rmse': []}
        for i in range(1, 2):
            if not baseline:
                print("X Shape = ")
                print(X.shape)

                # poly_features = X
                poly = PolynomialFeatures(degree=1)
                poly_features = poly.fit_transform(X)

                X_train, X_test, Y_train, Y_test = train_test_split(poly_features, Y, test_size=0.33, random_state=42)

                learning_rate_init = 0.001 / (1 * 30.0) # 30 yields best results
                epsilon = 1 * 10 ** -(2) # 1e-2 yields best results
                # epsilon = 1e-8
                clf = MLPRegressor(random_state=1, max_iter=50000, learning_rate_init=learning_rate_init, epsilon=epsilon)
                clf.fit(X_train, Y_train.values.ravel()) # Y_train.values.ravel()
                Y_pred = clf.predict(X_test)


                results['clf'].append(clf)
                results['X_train'].append(X_train)
                results['Y_pred'].append(Y_pred)
                results['Y_test'].append(Y_test)

                mse = mean_squared_error(Y_test, Y_pred)
                rmse = mean_squared_error(Y_test, Y_pred, squared=False)
            else:
                mse = mean_squared_error(Y, Y_baseline)
                rmse = mean_squared_error(Y, Y_baseline, squared=False)

            results['x'].append(i)
            results['mse'].append(mse)
            results['rmse'].append(rmse)
        return results

    def plotResults(results, title):
        """plt.title(title)
        plt.plot(results['x'], results['mse'], label = "MSE")
        plt.plot(results['x'], results['rmse'], label = "RMSE")
        plt.legend()
        plt.show()"""
        print(title)
        print("MSEs: {}".format(results['mse']))
        print("RMSEs: {}".format(results['rmse']))
        print()


from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer

class NeuralNetClassificationModel():
    def getYClassificationFromMerged(merged):
        Y_clf = merged.apply(lambda row: NeuralNetClassificationModel.classifyGrowth(row['percentage']), axis=1)
        return Y_clf

    def getYBaselineClassificationFromMerged(merged):
        baselinePercentage = .06
        Y_clf_baseline = merged.apply(lambda row: NeuralNetClassificationModel.classifyGrowth(baselinePercentage), axis=1)
        return Y_clf_baseline

    def classifyGrowth(percentage):
        growthClassification = 0

        if percentage < 0:
            growthClassification = -1
        elif percentage > 0:
            growthClassification = 1

        return growthClassification

    def declassifyGrowth(growthClassification):
        baselinePercentage = .06
        percentage = growthClassification * baselinePercentage

        if growthClassification == -1:
            percentage = 0.0
        # elif growthClassification == 0:
        #     percentage = 0
        # elif growthClassification == 1:
        # pass

        return percentage


    def trainAndGetResults(X, Y, Y_baseline, baseline=False):
        results = {'clf': [] , 'x': [], 'X_train': [], 'Y_pred': [], 'Y_test': [], 'mse': [], 'rmse': []}
        for i in range(1, 2):
            if not baseline:
                print("X Shape = ")
                print(X.shape)

                # poly_features = X
                poly = PolynomialFeatures(degree=1)
                poly_features = poly.fit_transform(X)

                X_train, X_test, Y_train, Y_test = train_test_split(poly_features, Y, test_size=0.33, random_state=42)

                clf = NeuralNetClassificationModel.trainModel(X_train, Y_train)
                Y_pred = clf.predict(X_test)
                # print(classification_report(Y_test, clf))

                results['clf'].append(clf)
                results['X_train'].append(X_train)
                results['Y_pred'].append(Y_pred)
                results['Y_test'].append(Y_test)

                mse = mean_squared_error(Y_test, Y_pred)
                rmse = mean_squared_error(Y_test, Y_pred, squared=False)
            else:
                mse = mean_squared_error(Y, Y_baseline)
                rmse = mean_squared_error(Y, Y_baseline, squared=False)

            results['x'].append(i)
            results['mse'].append(mse)
            results['rmse'].append(rmse)
        return results

    def trainModel(X_train, Y_train):
        """
        Returns the model currently used in the classifier.
        """
        return NeuralNetClassificationModel.trainGridSearchModel(X_train, Y_train)

    def trainUntunedModel(X_train, Y_train):
        clf = MLPClassifier(random_state=1, max_iter=5000)
        clf.fit(X_train, Y_train)
        # Y_pred = clf.predict_proba(X_test)
        # Y_pred = clf.predict(X_test)
        return clf

    def trainGridSearchModel(X_train, Y_train):
        """
        Current Best Params:
        {'alpha': 1e-05, 'epsilon': 1e-08, 'learning_rate_init': 0.05}
        {'alpha': 0.01, 'epsilon': 10, 'learning_rate_init': 0.05}
        """

        # clf = NeuralNetClassificationModel.trainUntunedModel(X_train, Y_train)
        clf = MLPClassifier(random_state=1, max_iter=5000)

        """
        grid_search = {'learning_rate_init': [0.001 / 60, 0.001 / 30, 0.01, 0.05, 0.5],
                       'epsilon': [1e-14, 1e-8, 1e-2, 10],
                       'alpha': list(np.linspace(.01, .00001, num=5, dtype = float)) } # .0001 is default
        """

        # alpha is the regularization constant
        grid_search = {'learning_rate_init': [0.05],
                       'epsilon': [1e-08],
                       'hidden_layer_sizes': [(20, 20), (20,), (100,), (20, 5), (5, 20), (100, 20, 5)],
                       'alpha': 10.0 ** -np.arange(1, 7) } # .0001 is default
                       # 'alpha': [1e-05] }


        mseScorer = make_scorer(mean_squared_error, greater_is_better=False)

        model = GridSearchCV(estimator = clf, param_grid = grid_search, scoring = mseScorer, n_jobs=-1, cv=3)
        model.fit(X_train, Y_train)

        print(model.best_params_)
        # print(gridResult)
        # Y_pred = model.best_estimator_.predict(X_test)
        clf = model.best_estimator_

        # print(classification_report(Y_test, clf))

        return clf

    def trainRandomSearchModel(X_train, Y_train):
        # random_search = {'learning_rate_init': list(np.linspace(10, 1200, num=10, dtype = float))}
        random_search = {'learning_rate_init': [0.001 / 60, 0.001 / 30, 0.001 / 1, 0.01, 0.05, 0.1, 0.5],
                       'epsilon': [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-0, 10, 100]}

        model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 80,
                                       cv = 4, verbose= 5, random_state= 101, n_jobs = -1)
        model.fit(X_train, Y_train)

        print(model.best_params_)
        # Y_pred = model.predict(X_test)
        clf = model
        return clf

    def predict(clf, X):
        """
        Converts the classification prediction into a real-valued prediction
        using the S&P 500 baseline percentage.
        """
        predictedGrowthClassifications = clf.predict(X)

        # the stuff below isn't actually that complicated and could easily be done with a for loop
        declassifier = np.vectorize(lambda growthClassification: NeuralNetClassificationModel.declassifyGrowth(growthClassification))
        predictedGrowthPercentages = declassifier(predictedGrowthClassifications)

        return predictedGrowthPercentages

    def plotResults(results, title):
        plt.title(title)
        plt.plot(results['x'], results['mse'], label = "MSE")
        plt.plot(results['x'], results['rmse'], label = "RMSE")
        plt.legend()
        plt.show()
        print(title)
        print("MSEs: {}".format(results['mse']))
        print("RMSEs: {}".format(results['rmse']))
        print()
