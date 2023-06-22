#############################################################################################
from predictive_models import *
##############################################################################################
results = dict()

groups = []

for i in range(7):
    for j in range(2048):
        groups.append(i)

test_subjects = [0, 1, 2, 4, 5, 6, 7]
##############################################################################################
# select the best models and graph the test accuracy value as the number of characteristics changes

result = features_selection_and_model_optimization("dataset_norm/dataset_norm_clean_balanced_with_updated_v_and_a_features_without_test_3.csv",
                                                   test_subjects=test_subjects, validation_size=0.0,
                                                   low_value=5, high_value=105, step=5,
                                                   project_dir="features_selection_and_optimization",
                                                   project_name=f"SVM_linear_svc", feature_selection_type='from_score_func',
                                                   model_type="SVM_linear_svc", groups=groups, train_and_test_n_iteration=10,
                                                   retest=False, class_labels=['NoEngagement', 'Engagement'])

results[f'SVM_linear_svc'] = result


result = features_selection_and_model_optimization("dataset_norm/dataset_norm_clean_balanced_with_updated_v_and_a_features_without_test_3.csv",
                                                   test_subjects=test_subjects, validation_size=0.0,
                                                   low_value=5, high_value=105, step=5,
                                                   project_dir="features_selection_and_optimization",
                                                   project_name=f"SVM_poly_svc", feature_selection_type='from_score_func',
                                                   model_type="SVM_poly_svc", groups=groups, train_and_test_n_iteration=10,
                                                   retest=False, class_labels=['NoEngagement', 'Engagement'])

results[f'SVM_poly_svc'] = result

result = features_selection_and_model_optimization("dataset_norm/dataset_norm_clean_balanced_with_updated_v_and_a_features_without_test_3.csv",
                                                   test_subjects=test_subjects, validation_size=0.0,
                                                   low_value=5, high_value=105, step=5,
                                                   project_dir="features_selection_and_optimization",
                                                   project_name=f"SVM_rbf_svc", feature_selection_type='from_score_func',
                                                   model_type="SVM_rbf_svc", groups=groups, train_and_test_n_iteration=10,
                                                   retest=False, class_labels=['NoEngagement', 'Engagement'])

results[f'SVM_rbf_rbf_svc'] = result

result = features_selection_and_model_optimization("dataset_norm/dataset_norm_clean_balanced_with_updated_v_and_a_features_without_test_3.csv",
                                                   test_subjects=test_subjects, validation_size=0.0,
                                                   low_value=5, high_value=105, step=5,
                                                   project_dir="features_selection_and_optimization",
                                                   project_name=f"SVM_linear_svc_2", feature_selection_type='from_score_func',
                                                   model_type="SVM_linear_svc_2", groups=groups, train_and_test_n_iteration=10,
                                                   retest=False, class_labels=['NoEngagement', 'Engagement'])

results[f'SVM_linear_svc_2'] = result

result = features_selection_and_model_optimization("dataset_norm/dataset_norm_clean_balanced_with_updated_v_and_a_features_without_test_3.csv",
                                                   test_subjects=test_subjects, validation_size=0.0,
                                                   low_value=5, high_value=105, step=5,
                                                   project_dir="features_selection_and_optimization",
                                                   project_name=f"DECISION_TREE",
                                                   feature_selection_type='from_score_func', model_type="DECISION_TREE",
                                                   groups=groups, train_and_test_n_iteration=10,
                                                   retest=False, class_labels=['NoEngagement', 'Engagement'])

results[f'DECISION_TREE'] = result

result = features_selection_and_model_optimization("dataset_norm/dataset_norm_clean_balanced_with_updated_v_and_a_features_without_test_3.csv",
                                                   test_subjects=test_subjects, validation_size=0.0,
                                                   low_value=5, high_value=105, step=5,
                                                   project_dir="features_selection_and_optimization",
                                                   project_name=f"RANDOM_FOREST",
                                                   feature_selection_type='from_score_func', model_type="RANDOM_FOREST",
                                                   groups=groups, train_and_test_n_iteration=10,
                                                   retest=False, class_labels=['NoEngagement', 'Engagement'])

results[f'RANDOM_FOREST'] = result

result = features_selection_and_model_optimization("dataset_norm/dataset_norm_clean_balanced_with_updated_v_and_a_features_without_test_3.csv",
                                                   test_subjects=test_subjects, validation_size=0.0,
                                                   low_value=5, high_value=105, step=5,
                                                   project_dir="features_selection_and_optimization",
                                                   project_name=f"KNN",
                                                   feature_selection_type='from_score_func', model_type="KNN",
                                                   groups=groups, train_and_test_n_iteration=10,
                                                   retest=False, class_labels=['NoEngagement', 'Engagement'])

results[f'KNN'] = result

result = features_selection_and_model_optimization("dataset_norm/dataset_norm_clean_balanced_with_updated_v_and_a_features_without_test_3.csv",
                                                   test_subjects=test_subjects, validation_size=0.3,
                                                   low_value=5, high_value=105, step=5,
                                                   project_dir="features_selection_and_optimization",
                                                   project_name=f"MLP",
                                                   feature_selection_type='from_score_func', model_type="MLP",
                                                   groups=groups, train_and_test_n_iteration=5,
                                                   retest=False, class_labels=['NoEngagement', 'Engagement'])

results[f'MLP'] = result


try:
    os.mkdir("features_selection_and_optimization/all_models")
except FileExistsError:
    pass

with open(f"features_selection_and_optimization/all_models/experiments.json", "w") as outfile:
    json.dump(results, outfile)

