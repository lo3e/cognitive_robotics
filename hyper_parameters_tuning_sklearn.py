from predictive_models import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import uniform


def model_parameters_grid_optimization(model, X, y, parameters, groups=None):

    logo = LeaveOneGroupOut()

    if groups:
        clf = GridSearchCV(model, parameters, cv=logo, verbose=3, n_jobs=6)
        search = clf.fit(X, y, groups=groups)
    else:
        clf = GridSearchCV(model, parameters, verbose=3, n_jobs=6)
        search = clf.fit(X, y)

    return {'results': clf.cv_results_, 'best_estimator': clf.best_estimator_, 'best_params': clf.best_params_,
            'best_index': clf.best_index_, 'best_score': clf.best_score_}


def model_parameters_randomized_optimization(model, X, y, parameters_distributions, groups=None):

    logo = LeaveOneGroupOut()

    clf = RandomizedSearchCV(model, parameters_distributions, cv=logo, verbose=3, random_state=0, n_jobs=12)
    search = clf.fit(X, y, groups=groups)

    return {'results': clf.cv_results_, 'best_estimator': clf.best_estimator_, 'best_params': clf.best_params_,
            'best_index': clf.best_index_, 'best_score': clf.best_score_}


def sklearn_model_optimization(Xtrain, ytrain, model_type, groups=None):

    result = None

    start_optimization_time = time.time()

    if model_type == 'SVM_linear_svc':
        # linear svc model
        model = svm.SVC()

        parameters = {'kernel': ['linear'], 'C': [0.1, 1, 10], 'tol': [1e-4, 1e-3, 1e-2],
                      'shrinking': [True, False]}
        result = model_parameters_grid_optimization(model, Xtrain, ytrain, parameters, groups=groups)

        #parameters_distributions = dict(C=uniform(loc=0, scale=10))

        #model_parameters_randomized_optimization(model, Xtrain, ytrain, parameters_distributions)

        print(result['best_params'])

    if model_type == 'SVM_rbf_svc':
        # linear rbf model
        model = svm.SVC()

        parameters = {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'],
                      'tol': [1e-4, 1e-3, 1e-2], 'shrinking': [True, False]}

        result = model_parameters_grid_optimization(model, Xtrain, ytrain, parameters, groups=groups)
        print(result['best_params'])

        #parameters_distributions = dict(C=uniform(loc=0, scale=10), degree=uniform(loc=0, scale=10), gamma=['scale', 'auto'])

        #model_parameters_randomized_optimization(model, Xtrain, ytrain, parameters_distributions)

    if model_type == 'SVM_poly_svc':
        # linear poly model
        model = svm.SVC()

        parameters = {'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [2, 3, 4],
                      'gamma': ['scale', 'auto'], 'coef0': [0], 'tol': [1e-4, 1e-3, 1e-2],
                      'shrinking': [True, False]}

        result = model_parameters_grid_optimization(model, Xtrain, ytrain, parameters, groups=groups)
        print(result['best_params'])

        #parameters_distributions = dict(C=uniform(loc=0, scale=10), degree=uniform(loc=0, scale=10), gamma=['scale', 'auto'],
        #                                coef0=uniform(loc=0, scale=10))

        #model_parameters_randomized_optimization(model, Xtrain, ytrain, parameters_distributions)

    if model_type == 'SVM_linear_svc_2':
        # linear svc second model
        model = svm.LinearSVC()

        parameters = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'],  'dual': [False],
                      'loss': ['hinge', 'squared_hinge'], 'max_iter': [1000, 2000], 'tol': [1e-5, 1e-4, 1e-3]}

        result = model_parameters_grid_optimization(model, Xtrain, ytrain, parameters, groups=groups)
        print(result['best_params'])

        #parameters_distributions = dict(C=uniform(loc=0, scale=10), dual=[True, False], max_iter=[1000, 10000])

        #model_parameters_randomized_optimization(model, Xtrain, ytrain, parameters_distributions)

    if model_type == 'DECISION_TREE':
        # decision tree model
        model = DecisionTreeClassifier()

        parameters = {'criterion': ['entropy'], 'splitter': ['best', 'random'], 'max_depth': [3, 6, 9, 12, 15, None],
                      'min_samples_split': [2, 3, 4, 5], 'min_samples_leaf': [1, 2, 3, 4],
                      'max_features': ['auto', 'sqrt', 'log2', None], 'min_impurity_decrease': [0.0],
                      'ccp_alpha': [0.0, 0.010, 0.020, 0.030]}


        result = model_parameters_grid_optimization(model, Xtrain, ytrain, parameters, groups=groups)
        print(result['best_params'])

        #parameters_distributions = dict(splitter=['best', 'random'], max_depth=uniform(loc=1, scale=50),
        #                                min_samples_split=uniform(loc=2, scale=50), min_samples_leaf=uniform(loc=1, scale=50),
        #                                min_weight_fraction_leaf=uniform(loc=0.0, scale=1.0), max_features=['auto', 'sqrt', 'log2'],
        #                                ccp_alpha=uniform(loc=0.0, scale=10.0))

        #model_parameters_randomized_optimization(model, Xtrain, ytrain, parameters_distributions)

    if model_type == 'RANDOM_FOREST':
        # random forest model
        # (fisso i parametri ottimale del singolo albero decisionale e ottimizzo solo i parametri intrinsechi del modello "random forest")
        model = RandomForestClassifier(random_state=0, n_jobs=-1)

        parameters = {'criterion': ['entropy'], 'max_depth': [3, None], 'min_samples_split': [2],
                      'min_samples_leaf': [1], 'max_features': [None],
                      'n_estimators': [50, 100, 200], 'bootstrap': [True, False], 'min_impurity_decrease': [0.0],
                      'oob_score': [True, False], 'warm_start': [True, False], 'max_samples': [None],
                      'ccp_alpha': [0.0]}

        result = model_parameters_grid_optimization(model, Xtrain, ytrain, parameters, groups=groups)
        print(f"best parameters: {result['best_params']}")
        print(f"best score: {result['best_score']}")

        #parameters_distributions = dict(n_estimators=uniform(loc=10, scale=1000), bootstrap=[True, False], oob_score=[True, False],
        #                                warm_start=[True, False], max_samples=uniform(loc=0.0, scale=1.0))

        #model_parameters_randomized_optimization(model, Xtrain, ytrain, parameters_distributions)

    if model_type == 'KNN':

        model = KNeighborsClassifier(n_jobs=-1)

        parameters = {'n_neighbors': [5, 10, 20], 'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'leaf_size': [30, 60, 120], 'p': [1, 2, 3]}

        result = model_parameters_grid_optimization(model, Xtrain, ytrain, parameters, groups=groups)
        print(f"best parameters: {result['best_params']}")
        print(f"best score: {result['best_score']}")

    stop_optimization_time = time.time()

    optimization_time = stop_optimization_time - start_optimization_time

    result['optimization_time'] = optimization_time

    return result


