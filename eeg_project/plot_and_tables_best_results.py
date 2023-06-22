import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

best_results = json.load(open("features_selection_and_optimization/all_models/best_results.json"))

plot = plt.plot()

y_labels = ["svc linear", "svc poly", "svc rbf", "svc linear 2", "decision tree", "random forest", "knn", 'mlp']


x = [i for i in range(5, 105, 5)]
x_label = 'percentile'
labels = []
for key in best_results:
    #y_labels.append(f'{key}')
    y = best_results[key]['average_accuracy']
    plt.plot(x, y)

y_labels = tuple(y_labels)
plt.gca().legend(y_labels, loc='best', bbox_to_anchor=(1.05, 1))
plt.tight_layout(pad=2)

plt.title("average accuracy vs percentile")
plt.xlabel(x_label)
plt.ylabel("accuracy")
plt.grid()

plt.savefig("features_selection_and_optimization/all_models/accuracy_vs_percentile.png")

plt.close()


plot = plt.plot()

#y_labels = []


x = [i for i in range(5, 105, 5)]
x_label = 'percentile'

for key in best_results:
    #y_labels.append(f'{key}')
    y = best_results[key]['average_precision']
    plt.plot(x, y)

y_labels = tuple(y_labels)
plt.gca().legend(y_labels, loc='best', bbox_to_anchor=(1.05, 1))
plt.tight_layout(pad=2)

plt.title("average precision vs percentile")
plt.xlabel(x_label)
plt.ylabel("precision")
plt.grid()

plt.savefig("features_selection_and_optimization/all_models/precision_vs_percentile.png")

plt.close()

plot = plt.plot()

y_labels = []


x = [i for i in range(5, 105, 5)]
x_label = 'percentile'

for key in best_results:
    y = np.array(best_results[key]['average_precision_0']) - np.array(best_results[key]['average_precision_1'])
    plt.plot(x, y)

y_labels = tuple(y_labels)
plt.gca().legend(y_labels, loc='best', bbox_to_anchor=(1.05, 1))
plt.tight_layout(pad=2)

plt.title("average delta precision vs percentile")
plt.xlabel(x_label)
plt.ylabel("delta precision")
plt.grid()

plt.savefig("features_selection_and_optimization/all_models/delta_precision_vs_percentile.png")

plt.close()

########################################################################################################################
column_labels = ['percentile', 'accuracy', 'avg_precision', 'avg_recall', 'avg_f1_score', 'precision_0', 'recall_0',
          'f1_score_0', 'precision_1', 'recall_1', 'f1_score_1', 'prediction_time (s)']

index_labels = ["svc linear", "svc poly", "svc rbf", "svc linear 2", "mlp", "knn", "random forest", "decision tree"]

best_metrics_table_to_image = []

# decision tree
best_tree = [15,0.8443,0.86062,0.8443,0.8393,0.8637,0.8245,0.8284,0.8576,0.864,0.8502,np.format_float_scientific(7.2e-08, precision=2, exp_digits=2)]
# KNN
best_knn = [50,0.9057,0.91194,0.9057,0.9046,0.8858,0.9445,0.9116,0.9381,0.8669,0.8975,np.format_float_scientific(1.7e-04, precision=2, exp_digits=2)]
# RANDOM FOREST
best_random_forest = [5,0.8893,0.90631,0.8893,0.8872,0.8863,0.91,0.8884,0.9263,0.8686,0.8861,np.format_float_scientific(3.8e-06, precision=2, exp_digits=2)]
#SVM_linear_svc
best_svc_linear = [5,0.9223,0.9257,0.9223,0.922,0.9214,0.9275,0.9222,0.93,0.9171,0.9218,np.format_float_scientific(2.0e-05, precision=2, exp_digits=2)]
#SVM_poly_svc
best_svc_linear_2 = [5,0.9177,0.92302,0.9177,0.9171,0.9093,0.9353,0.9188,0.9368,0.9,0.9154,np.format_float_scientific(1.3e-07, precision=2, exp_digits=2)]
#SVM_rbf_svc
best_svc_poly = [5,0.92,0.92361,0.92,0.9197,0.9048,0.9439,0.9224,0.9424,0.8961,0.917,np.format_float_scientific(1.5e-05, precision=2, exp_digits=2)]
#SVM_linear_svc
best_svc_rbf= [5,0.9185,0.92254,0.9185,0.9182,0.9085,0.9353,0.9196,0.9366,0.9018,0.9168,np.format_float_scientific(4.3e-05, precision=2, exp_digits=2)]
#MLP
best_mlp = [5,0.9136,0.91963,0.9136,0.9129,0.9113,0.9207,0.9122,0.928,0.9064,0.9137,np.format_float_scientific(2.9e-05, precision=2, exp_digits=2)]

best_metrics_table_to_image.append(np.array(best_svc_linear))
best_metrics_table_to_image.append(np.array(best_svc_poly))
best_metrics_table_to_image.append(np.array(best_svc_rbf))
best_metrics_table_to_image.append(np.array(best_svc_linear_2))
best_metrics_table_to_image.append(np.array(best_mlp))
best_metrics_table_to_image.append(np.array(best_knn))
best_metrics_table_to_image.append(np.array(best_random_forest))
best_metrics_table_to_image.append(np.array(best_tree))

best_metrics_table_to_image = np.array(best_metrics_table_to_image)

df = pd.DataFrame(best_metrics_table_to_image, index=index_labels, columns=column_labels)

import dataframe_image as dfi

dfi.export(df, "features_selection_and_optimization/all_models/best_models_metrics_table.png")
########################################################################################################################

