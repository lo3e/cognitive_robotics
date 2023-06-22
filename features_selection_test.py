from predictive_models import *


for function in (sklearn.feature_selection.f_classif,):

    experiment = dict()
    experiment['f_classif'] = {'type_percentile': {'selected_features_names': list(), 'scores': list(), 'pvalues': list()},
                               'type_Kbest': {'selected_features_names': list(), 'scores': list(), 'pvalues': list()}}

    for value in range(5, 105, 5):

        print(f"extracting dataset...")
        dataset = get_dataset_from_csv("dataset_norm/dataset_norm_clean.csv", [], 0.0)

        print(f"selecting with percentile = {value}")
        selected_features = features_selection_from_score_function(dataset['Xtrain'], dataset['ytrain'], dataset['Xval'],
                                                                   dataset['Xtest'], percentile=value,
                                                                   function=function,
                                                                   type='percentile')

        print(list(selected_features['selected_features_names']))
        print(list(selected_features['scores']))
        print(list(selected_features['pvalues']))

        experiment['f_classif']['type_percentile']['selected_features_names'].append(list(selected_features['selected_features_names']))
        experiment['f_classif']['type_percentile']['scores'].append(list(selected_features['scores']))
        experiment['f_classif']['type_percentile']['pvalues'].append(list(selected_features['pvalues']))

        print(f"selecting with Kbest = {int((value / 100) * len(dataset['Xtrain'][0]))}")
        selected_features = features_selection_from_score_function(dataset['Xtrain'], dataset['ytrain'],
                                                                   dataset['Xval'],
                                                                   dataset['Xtest'], percentile=value,
                                                                   function=function,
                                                                   type='Kbest')

        print(list(selected_features['selected_features_names']))
        print(list(selected_features['scores']))
        print(list(selected_features['pvalues']))

        experiment['f_classif']['type_Kbest']['selected_features_names'].append(list(selected_features['selected_features_names']))
        experiment['f_classif']['type_Kbest']['scores'].append(list(selected_features['scores']))
        experiment['f_classif']['type_Kbest']['pvalues'].append(list(selected_features['pvalues']))

try:
    os.mkdir("features_selection_and_optimization/all_models")
except FileExistsError:
    pass

with open(f"features_selection_and_optimization/features_selection_test.json", "w") as outfile:
    json.dump(experiment, outfile)
