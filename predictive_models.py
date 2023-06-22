##############################################################################################
##############################################################################################
import csv
import os
import json
from statistics import mean, stdev
import keras.models
import sklearn
import copy
import pandas as pd
import random
import time
from math import log
import numpy as np
from numpy import array
from sklearn import svm
from sklearn.model_selection import train_test_split as split
from sklearn import tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, SelectFromModel, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from hyper_parameters_tuning_keras import keras_model_optimization, model_builder
from hyper_parameters_tuning_sklearn import sklearn_model_optimization
import labels_generator
from function import scientific_notation
##############################################################################################
##############################################################################################


def get_dataset_from_csv(csv_file_path, test_subjects, validation_size, exclude_from_training=tuple()):

    with open(csv_file_path, newline="", encoding="ISO-8859-1") as filecsv:
        lettore = csv.reader(filecsv, delimiter=",")
        dataset = []
        for row in lettore:
            dataset.append(row)
    for row in dataset:
        for i in range(len(row)):
            row[i] = float(row[i])
        row[-1] = int(row[-1])


    splitted_dataset = split_by_subjects(dataset, test_subjects=test_subjects, validation_size=validation_size,
                                         exclude_from_training=exclude_from_training)

    Xtrain = splitted_dataset["Xtrain"]
    Xvalidation = splitted_dataset["Xvalidation"]
    Xtest = splitted_dataset["Xtest"]

    ytrain = []
    for elemento in Xtrain:
        ytrain.append(int(elemento[-1]))
    for i in range(len(Xtrain)):
        Xtrain[i] = Xtrain[i][:len(Xtrain[i]) - 1]

    yvalidation = []
    for elemento in Xvalidation:
        yvalidation.append(int(elemento[-1]))
    for i in range(len(Xvalidation)):
        Xvalidation[i] = Xvalidation[i][:len(Xvalidation[i]) - 1]

    ytest = []
    for elemento in Xtest:
        ytest.append(int(elemento[-1]))
    for i in range(len(Xtest)):
        Xtest[i] = Xtest[i][:len(Xtest[i]) - 1]

    Xtrain = np.array(Xtrain)
    Xvalidation = np.array(Xvalidation)
    Xtest = np.array(Xtest)
    ytrain = np.array(ytrain)
    yvalidation = np.array(yvalidation)
    ytest = np.array(ytest)

    # set dimension
    height_train = len(Xtrain)
    height_validation = len(Xvalidation)
    height_test = len(Xtest)
    width = len(Xtrain[0])
    channels = 1

    Xtrain = Xtrain.reshape(height_train, width, )
    Xvalidation = Xvalidation.reshape(height_validation, width, )
    Xtest = Xtest.reshape(height_test, width, )

    return {'Xtrain': Xtrain, 'Xval': Xvalidation, 'Xtest': Xtest,
            'ytrain': ytrain, 'yval': yvalidation, 'ytest': ytest}
##############################################################################################


def label_separation(dataset):
    labelset = []

    for i in range(len(dataset)):
        labelset.append(dataset[i][- 1])
        dataset[i] = dataset[i][:len(dataset[i]) - 1]

    return dataset, labelset
######################################################################################


def shuffle(X, Y):

    dataset = []

    for x, y in zip(X, Y):
        dataset.append((x, y))

    random.shuffle(dataset)

    new_X = []
    new_Y = []

    for x_y in dataset:
        new_X.append(x_y[0])
        new_Y.append(x_y[1])

    new_X = np.array(new_X)
    new_Y = np.array(new_Y)

    return new_X, new_Y
######################################################################################


def split_by_subjects(X, test_subjects, validation_size=0.3, balancing_train=False, balancing_test=False, shuffle=False,
                      exclude_from_training=()):
    Xtrain = []
    Xtest = []

    show=[]
    for row in X:
        if int(row[-1]) in test_subjects:
            Xtest.append(copy.deepcopy(row))
        elif int(row[-1]) not in exclude_from_training:
            Xtrain.append(copy.deepcopy(row))
        else:
            show.append(row[-1])
            pass

    if balancing_train:
        X_train_class0 = [Xtrain[i] for i in range(len(Xtrain)) if Xtrain[i][-2] == 0]
        X_train_class1 = [Xtrain[i] for i in range(len(Xtrain)) if Xtrain[i][-2] == 1]

        if len(X_train_class0) > len(X_train_class1):
            balanced_Xtrain = copy.deepcopy(X_train_class1)

            for i in range(len(X_train_class1)):
                random_index = random.randint(0, len(X_train_class0) - 1)
                balanced_Xtrain.append(X_train_class0.pop(random_index))

            Xtrain = balanced_Xtrain

        elif len(X_train_class0) < len(X_train_class1):
            balanced_Xtrain = copy.deepcopy(X_train_class0)

            for i in range(len(X_train_class0)):
                random_index = random.randint(0, len(X_train_class1) - 1)
                balanced_Xtrain.append(X_train_class1.pop(random_index))

            Xtrain = balanced_Xtrain

    if balancing_test:
        X_test_class0 = [Xtest[i] for i in range(len(Xtest)) if Xtest[i][-2] == 0]
        X_test_class1 = [Xtest[i] for i in range(len(Xtest)) if Xtest[i][-2] == 1]

        if len(X_test_class0) > len(X_test_class1):
            balanced_Xtest = copy.deepcopy(X_test_class1)

            for i in range(len(X_train_class1)):
                random_index = random.randint(0, len(X_train_class0) - 1)
                balanced_Xtest.append(X_train_class0.pop(random_index))

            Xtest = balanced_Xtest

        elif len(X_test_class0) < len(X_test_class1):
            balanced_Xtest = copy.deepcopy(X_test_class0)

            for i in range(len(X_test_class0)):
                random_index = random.randint(0, len(X_test_class1) - 1)
                balanced_Xtest.append(X_test_class1.pop(random_index))

            Xtest = balanced_Xtest

    if validation_size != 0.0:
        Xtrain, Xvalidation = split(Xtrain, test_size=validation_size)
    else:
        Xvalidation = []

    if shuffle:
        random.shuffle(Xtrain)
        random.shuffle(Xvalidation)
        random.shuffle(Xtest)

    for i in range(len(Xtrain)):
        Xtrain[i] = Xtrain[i][:-1]
    for i in range(len(Xvalidation)):
        Xvalidation[i] = Xvalidation[i][:-1]
    for i in range(len(Xtest)):
        Xtest[i] = Xtest[i][:-1]

    return {"Xtrain": Xtrain, "Xvalidation": Xvalidation, "Xtest": Xtest}
######################################################################################


def NN_plot(history, test_scores=None, image_file_path=None):

    # grafici di accuracy e loss per le fasi di train e test
    # l'asse delle ascisse corrisponde alle epoche trascors
    #######################################################
    # Plot training & validation loss values
    # Plot training & validation accuracy values
    plt.figure(figsize=(20, 10))
    # Grafico riga 1, colonna 1
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    if test_scores:
        plt.plot(test_scores['test_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    if test_scores:
        plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
    else:
        plt.legend(['Train', 'Validation'], loc='upper left')
    # Grafico riga 1, colonna 2
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    if test_scores:
        plt.plot(test_scores['test_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    if test_scores:
        plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
    else:
        plt.legend(['Train', 'Validation'], loc='upper left')

    if image_file_path:
        plt.savefig(image_file_path)

    plt.close()

    return plt
##############################################################################################


def plot_metrics_vs_selected_features(experiments, images_files_path, x_label, class_labels=None):

    model_key = list(experiments.keys())[0]

    x = experiments['steps']

    ####################################################################################################################
    accuracy_values = dict()
    precision = dict()
    recall = dict()
    f1_score = dict()
    precision_class_0 = dict()
    recall_class_0 = dict()
    f1_score_class_0 = dict()
    precision_class_1 = dict()
    recall_class_1 = dict()
    f1_score_class_1 = dict()

    accuracy_values['total'] = dict()
    precision['total'] = dict()
    recall['total'] = dict()
    f1_score['total'] = dict()
    precision_class_0['total'] = dict()
    recall_class_0['total'] = dict()
    f1_score_class_0['total'] = dict()
    precision_class_1['total'] = dict()
    recall_class_1['total'] = dict()
    f1_score_class_1['total'] = dict()
    ####################################################################################################################
    # print(experiments)

    for experiment in experiments[model_key]:
        ################################################################################################################
        if 'percentile' in experiment:
            x_value = experiment['percentile']
        elif 'threshold' in experiment:
            x_value = experiment['threshold']
        ################################################################################################################
        if f'{x_value}' not in accuracy_values['total']:
            accuracy_values['total'][f'{x_value}'] = list()

        if f'{x_value}' not in precision['total']:
            precision['total'][f'{x_value}'] = list()

        if f'{x_value}' not in recall['total']:
            recall['total'][f'{x_value}'] = list()

        if f'{x_value}' not in f1_score['total']:
            f1_score['total'][f'{x_value}'] = list()
        ################################################################################################################
        if f'{x_value}' not in precision_class_0['total']:
            precision_class_0['total'][f'{x_value}'] = list()

        if f'{x_value}' not in recall_class_0['total']:
            recall_class_0['total'][f'{x_value}'] = list()

        if f'{x_value}' not in f1_score_class_0['total']:
            f1_score_class_0['total'][f'{x_value}'] = list()
        ################################################################################################################
        if f'{x_value}' not in precision_class_1['total']:
            precision_class_1['total'][f'{x_value}'] = list()

        if f'{x_value}' not in recall_class_1['total']:
            recall_class_1['total'][f'{x_value}'] = list()

        if f'{x_value}' not in f1_score_class_1['total']:
            f1_score_class_1['total'][f'{x_value}'] = list()
        ################################################################################################################

        for report in experiment['report']:
            group = report['group']
            key = f'{group}'

            ############################################################################################################
            if key not in accuracy_values:
                accuracy_values[f'{group}'] = dict()
                accuracy_values[f'{group}'][f'{x_value}'] = list()
                accuracy_values[f'{group}'][f'{x_value}'].append(report['report']['accuracy'])
                accuracy_values['total'][f'{x_value}'].append(report['report']['accuracy'])
            else:
                if f'{x_value}' not in accuracy_values[f'{group}']:
                    accuracy_values[f'{group}'][f'{x_value}'] = list()
                accuracy_values[f'{group}'][f'{x_value}'].append(report['report']['accuracy'])
                accuracy_values['total'][f'{x_value}'].append(report['report']['accuracy'])
            ############################################################################################################
            if key not in precision:
                precision[f'{group}'] = dict()
                precision[f'{group}'][f'{x_value}'] = list()
                precision[f'{group}'][f'{x_value}'].append(report['report']['macro avg']['precision'])
                precision['total'][f'{x_value}'].append(report['report']['macro avg']['precision'])
            else:
                if f'{x_value}' not in precision[f'{group}']:
                    precision[f'{group}'][f'{x_value}'] = list()
                precision[f'{group}'][f'{x_value}'].append(report['report']['macro avg']['precision'])
                precision['total'][f'{x_value}'].append(report['report']['macro avg']['precision'])
            ############################################################################################################
            if key not in recall:
                recall[f'{group}'] = dict()
                recall[f'{group}'][f'{x_value}'] = list()
                recall[f'{group}'][f'{x_value}'].append(report['report']['macro avg']['recall'])
                recall['total'][f'{x_value}'].append(report['report']['macro avg']['recall'])
            else:
                if f'{x_value}' not in recall[f'{group}']:
                    recall[f'{group}'][f'{x_value}'] = list()
                recall[f'{group}'][f'{x_value}'].append(report['report']['macro avg']['recall'])
                recall['total'][f'{x_value}'].append(report['report']['macro avg']['recall'])
            ############################################################################################################
            if key not in f1_score:
                f1_score[f'{group}'] = dict()
                f1_score[f'{group}'][f'{x_value}'] = list()
                f1_score[f'{group}'][f'{x_value}'].append(report['report']['macro avg']['f1-score'])
                f1_score['total'][f'{x_value}'].append(report['report']['macro avg']['f1-score'])
            else:
                if f'{x_value}' not in f1_score[f'{group}']:
                    f1_score[f'{group}'][f'{x_value}'] = list()
                f1_score[f'{group}'][f'{x_value}'].append(report['report']['macro avg']['f1-score'])
                f1_score['total'][f'{x_value}'].append(report['report']['macro avg']['f1-score'])
            ############################################################################################################
            if key not in precision_class_0:
                precision_class_0[f'{group}'] = dict()
                precision_class_0[f'{group}'][f'{x_value}'] = list()
                precision_class_0[f'{group}'][f'{x_value}'].append(report['report'][class_labels[0]]['precision'])
                precision_class_0['total'][f'{x_value}'].append(report['report'][class_labels[0]]['precision'])
            else:
                if f'{x_value}' not in precision_class_0[f'{group}']:
                    precision_class_0[f'{group}'][f'{x_value}'] = list()
                precision_class_0[f'{group}'][f'{x_value}'].append(report['report'][class_labels[0]]['precision'])
                precision_class_0['total'][f'{x_value}'] .append(report['report'][class_labels[0]]['precision'])
            ############################################################################################################
            if key not in recall_class_0:
                recall_class_0[f'{group}'] = dict()
                recall_class_0[f'{group}'][f'{x_value}'] = list()
                recall_class_0[f'{group}'][f'{x_value}'].append(report['report'][class_labels[0]]['recall'])
                recall_class_0['total'][f'{x_value}'].append(report['report'][class_labels[0]]['recall'])
            else:
                if f'{x_value}' not in recall_class_0[f'{group}']:
                    recall_class_0[f'{group}'][f'{x_value}'] = list()
                recall_class_0[f'{group}'][f'{x_value}'].append(report['report'][class_labels[0]]['recall'])
                recall_class_0['total'][f'{x_value}'] .append(report['report'][class_labels[0]]['recall'])
            ############################################################################################################
            if key not in f1_score_class_0:
                f1_score_class_0[f'{group}'] = dict()
                f1_score_class_0[f'{group}'][f'{x_value}'] = list()
                f1_score_class_0[f'{group}'][f'{x_value}'].append(report['report'][class_labels[0]]['f1-score'])
                f1_score_class_0['total'][f'{x_value}'].append(report['report'][class_labels[0]]['f1-score'])
            else:
                if f'{x_value}' not in f1_score_class_0[f'{group}']:
                    f1_score_class_0[f'{group}'][f'{x_value}'] = list()
                f1_score_class_0[f'{group}'][f'{x_value}'].append(report['report'][class_labels[0]]['f1-score'])
                f1_score_class_0['total'][f'{x_value}'] .append(report['report'][class_labels[0]]['f1-score'])
            ############################################################################################################
            if key not in precision_class_1:
                precision_class_1[f'{group}'] = dict()
                precision_class_1[f'{group}'][f'{x_value}'] = list()
                precision_class_1[f'{group}'][f'{x_value}'].append(report['report'][class_labels[1]]['precision'])
                precision_class_1['total'][f'{x_value}'].append(report['report'][class_labels[1]]['precision'])
            else:
                if f'{x_value}' not in precision_class_1[f'{group}']:
                    precision_class_1[f'{group}'][f'{x_value}'] = list()
                precision_class_1[f'{group}'][f'{x_value}'].append(report['report'][class_labels[1]]['precision'])
                precision_class_1['total'][f'{x_value}'] .append(report['report'][class_labels[1]]['precision'])
            ############################################################################################################
            if key not in recall_class_1:
                recall_class_1[f'{group}'] = dict()
                recall_class_1[f'{group}'][f'{x_value}'] = list()
                recall_class_1[f'{group}'][f'{x_value}'].append(report['report'][class_labels[1]]['recall'])
                recall_class_1['total'][f'{x_value}'].append(report['report'][class_labels[1]]['recall'])
            else:
                if f'{x_value}' not in recall_class_1[f'{group}']:
                    recall_class_1[f'{group}'][f'{x_value}'] = list()
                recall_class_1[f'{group}'][f'{x_value}'].append(report['report'][class_labels[1]]['recall'])
                recall_class_1['total'][f'{x_value}'] .append(report['report'][class_labels[1]]['recall'])
            ############################################################################################################
            if key not in f1_score_class_1:
                f1_score_class_1[f'{group}'] = dict()
                f1_score_class_1[f'{group}'][f'{x_value}'] = list()
                f1_score_class_1[f'{group}'][f'{x_value}'].append(report['report'][class_labels[1]]['f1-score'])
                f1_score_class_1['total'][f'{x_value}'].append(report['report'][class_labels[1]]['f1-score'])
            else:
                if f'{x_value}' not in f1_score_class_1[f'{group}']:
                    f1_score_class_1[f'{group}'][f'{x_value}'] = list()
                f1_score_class_1[f'{group}'][f'{x_value}'].append(report['report'][class_labels[1]]['f1-score'])
                f1_score_class_1['total'][f'{x_value}'] .append(report['report'][class_labels[1]]['f1-score'])
            ############################################################################################################

    ####################################################################################################################
    average_accuracy_values = list()
    average_precision = list()
    average_recall = list()
    average_f1_score = list()
    average_precision_class_0 = list()
    average_recall_class_0 = list()
    average_f1_score_class_0 = list()
    average_precision_class_1 = list()
    average_recall_class_1 = list()
    average_f1_score_class_1 = list()

    accuracy_values_per_test = dict()
    precision_per_test = dict()
    recall_per_test = dict()
    f1_score_per_test = dict()
    precision_class_0_per_test = dict()
    recall_class_0_per_test = dict()
    f1_score_class_0_per_test = dict()
    precision_class_1_per_test = dict()
    recall_class_1_per_test = dict()
    f1_score_class_1_per_test = dict()

    for key in accuracy_values:
        for x_value in x:

            if key == 'total':
                average_accuracy_values.append(mean(accuracy_values[key][f'{x_value}']))
                average_precision.append(mean(precision[key][f'{x_value}']))
                average_recall.append(mean(recall[key][f'{x_value}']))
                average_f1_score.append(mean(f1_score[key][f'{x_value}']))
                average_precision_class_0.append(mean(precision_class_0[key][f'{x_value}']))
                average_recall_class_0.append(mean(recall_class_0[key][f'{x_value}']))
                average_f1_score_class_0.append(mean(f1_score_class_0[key][f'{x_value}']))
                average_precision_class_1.append(mean(precision_class_1[key][f'{x_value}']))
                average_recall_class_1.append(mean(recall_class_1[key][f'{x_value}']))
                average_f1_score_class_1.append(mean(f1_score_class_1[key][f'{x_value}']))

            else:
                if key not in accuracy_values_per_test:
                    accuracy_values_per_test[key] = list()

                accuracy_values_per_test[key].append(mean(accuracy_values[key][f'{x_value}']))
                if key not in precision_per_test:
                    precision_per_test[key] = list()

                precision_per_test[key].append(mean(precision[key][f'{x_value}']))
                if key not in recall_per_test:
                    recall_per_test[key] = list()

                recall_per_test[key].append(mean(recall[key][f'{x_value}']))
                if key not in f1_score_per_test:
                    f1_score_per_test[key] = list()

                f1_score_per_test[key].append(mean(f1_score[key][f'{x_value}']))
                if key not in precision_class_0_per_test:
                    precision_class_0_per_test[key] = list()

                precision_class_0_per_test[key].append(mean(precision_class_0[key][f'{x_value}']))
                if key not in recall_class_0_per_test:
                    recall_class_0_per_test[key] = list()

                recall_class_0_per_test[key].append(mean(recall_class_0[key][f'{x_value}']))
                if key not in f1_score_class_0_per_test:
                    f1_score_class_0_per_test[key] = list()

                f1_score_class_0_per_test[key].append(mean(f1_score_class_0[key][f'{x_value}']))
                if key not in precision_class_1_per_test:
                    precision_class_1_per_test[key] = list()

                precision_class_1_per_test[key].append(mean(precision_class_1[key][f'{x_value}']))
                if key not in recall_class_1_per_test:
                    recall_class_1_per_test[key] = list()

                recall_class_1_per_test[key].append(mean(recall_class_1[key][f'{x_value}']))
                if key not in f1_score_class_1_per_test:
                    f1_score_class_1_per_test[key] = list()

                f1_score_class_1_per_test[key].append(mean(f1_score_class_1[key][f'{x_value}']))

    average_accuracy_values = np.array(average_accuracy_values)
    average_precision = np.array(average_precision)
    average_recall = np.array(average_recall)
    average_f1_score = np.array(average_f1_score)
    average_precision_class_0 = np.array(average_precision_class_0)
    average_recall_class_0 = np.array(average_recall_class_0)
    average_f1_score_class_0 = np.array(average_f1_score_class_0)
    average_precision_class_1 = np.array(average_precision_class_1)
    average_recall_class_1 = np.array(average_recall_class_1)
    average_f1_score_class_1 = np.array(average_f1_score_class_1)

    for key in accuracy_values_per_test:
        accuracy_values_per_test[key] = np.array(accuracy_values_per_test[key])
        precision_per_test[key] = np.array(precision_per_test[key])
        recall_per_test[key] = np.array(recall_per_test[key])
        f1_score_per_test[key] = np.array(f1_score_per_test[key])
        precision_class_0_per_test[key] = np.array(precision_class_0_per_test[key])
        recall_class_0_per_test[key] = np.array(recall_class_0_per_test[key])
        f1_score_class_0_per_test[key] = np.array(f1_score_class_0_per_test[key])
        precision_class_1_per_test[key] = np.array(precision_class_1_per_test[key])
        recall_class_1_per_test[key] = np.array(recall_class_1_per_test[key])
        f1_score_class_1_per_test[key] = np.array(f1_score_class_1_per_test[key])

    ####################################################################################################################

    table_labels = ['percentile', 'accuracy', 'avg_precision', 'avg_recall', 'avg_f1_score', 'precision_0', 'recall_0',
                    'f1_score_0', 'precision_1', 'recall_1', 'f1_score_1', 'prediction_time (s)']

    metrics_table_to_csv = [table_labels]
    metrics_table_to_image = []

    for i in range(len(experiments['steps'])):
        metrics = []
        metrics.append(experiments['steps'][i])
        metrics.append(round(average_accuracy_values[i], 4))
        metrics.append(round(average_precision[i], 5))
        metrics.append(round(average_recall[i], 4))
        metrics.append(round(average_f1_score[i], 4))
        metrics.append(round(average_precision_class_0[i], 4))
        metrics.append(round(average_recall_class_0[i], 4))
        metrics.append(round(average_f1_score_class_0[i], 4))
        metrics.append(round(average_precision_class_1[i], 4))
        metrics.append(round(average_recall_class_1[i], 4))
        metrics.append(round(average_f1_score_class_1[i], 4))
        metrics.append((np.format_float_scientific(experiments[model_key][i]['prediction_time'], precision=1, exp_digits=2)))

        metrics_table_to_csv.append(metrics)
        metrics_table_to_image.append(np.array(metrics[1:]))

    metrics_table_to_image = np.array(metrics_table_to_image)

    df = pd.DataFrame(metrics_table_to_image, index=experiments['steps'], columns=table_labels[1:])

    f = open(f"{images_files_path}/metrics_table.csv", 'w')

    for i in range(len(metrics_table_to_csv)):
        for j in range(len(metrics_table_to_csv[i]) - 1):
            f.write(str(metrics_table_to_csv[i][j]) + ",")
        f.write(str(metrics_table_to_csv[i][-1]) + "\n")
    f.close()

    print(df)

    import dataframe_image as dfi
    dfi.export(df, f"{images_files_path}/metrics_table.png")

    ####################################################################################################################
    table_labels = ['percentile']
    for key in experiments[model_key][0]['best_params']:
        table_labels.append(key)

    params_table_to_image = []
    params_table_to_csv = [table_labels]
    for experiment in experiments[model_key]:
        params = [experiment['percentile']]
        for key in experiment['best_params']:
            if key == 'drop_values':
                params.append([round(z, 2) for z in experiment['best_params'][key]])
            else:
                params.append(experiment['best_params'][key])

        params_table_to_image.append(np.array(params[1:]))
        params_table_to_csv.append(params)

    params_table_to_image = np.array(params_table_to_image)

    df = pd.DataFrame(params_table_to_image, index=experiments['steps'], columns=table_labels[1:])

    f = open(f"{images_files_path}/params_table.csv", 'w')

    for i in range(len(params_table_to_csv)):
        for j in range(len(params_table_to_csv[i]) - 1):
            f.write(str(params_table_to_csv[i][j]) + ",")
        f.write(str(params_table_to_csv[i][-1]) + "\n")
    f.close()

    print(df)

    dfi.export(df, f"{images_files_path}/params_table.png")
    ####################################################################################################################

    files_list = os.listdir("features_selection_and_optimization/all_models")

    if "best_results.json" in files_list:
        results = json.load(open("features_selection_and_optimization/all_models/best_results.json"))
    else:
        results = dict()
    if model_key not in results:
        results[model_key] = dict()

    results[model_key]['average_accuracy'] = list(average_accuracy_values)
    results[model_key]['average_precision'] = list(average_precision)
    results[model_key]['average_precision_0'] = list(average_precision_class_0)
    results[model_key]['average_precision_1'] = list(average_precision_class_1)

    with open("features_selection_and_optimization/all_models/best_results.json", "w") as outfile:
        json.dump(results, outfile)

    ####################################################################################################################
    y = list()

    #for x_value in x:
    #    y.append(mean(accuracy_values['total'][f'{x_value}']))

    y = average_accuracy_values

    plot1 = plt.plot()
    plt.title("Accuracy vs percentile")
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.plot(x, y)
    plt.grid()

    plt.savefig(images_files_path + "/average_accuracy_vs_percentile.png")

    plt.close()

    plot2 = plt.plot()

    y_labels = []

    #for key in accuracy_values:
    #    y_labels.append(f'test on {key}')
    #    y = list()
    #    for x_value in x:
    #        y.append(mean(accuracy_values[key][f'{x_value}']))
    #    plt.plot(x, y)

    y_labels.append(f'test on total')
    y = average_accuracy_values
    plt.plot(x, y)
    for i in range(10):###########################################################################################
        for key in accuracy_values_per_test:
            if int(key) == i:
                y_labels.append(f'test on {key}')
                y = accuracy_values_per_test[key]
                plt.plot(x, y)

    y_labels = tuple(y_labels)
    plt.gca().legend(y_labels, loc='best', bbox_to_anchor=(1.05, 1))
    plt.tight_layout(pad=2)

    plt.title("accuracy vs percentile")
    plt.xlabel(x_label)
    plt.ylabel("accuracy")
    plt.grid()

    plt.savefig(images_files_path + "/accuracy_vs_percentile.png")

    plt.close()

    #y_precision_class_0 = list()
    #y_precision_class_1 = list()

    #for x_value in x:
    #    y_precision_class_0.append(mean(precision_class_0['total'][f'{x_value}']))
    #    y_precision_class_1.append(mean(precision_class_1['total'][f'{x_value}']))

    y_precision_class_0 = average_precision_class_0
    y_precision_class_1 = average_precision_class_1

    plot3 = plt.plot()
    plt.title("average precision vs percentile")
    plt.xlabel(x_label)
    plt.ylabel("precision")
    plt.plot(x, y_precision_class_0)
    plt.plot(x, y_precision_class_1)

    y_labels = class_labels
    plt.gca().legend(y_labels, loc='best')
    plt.tight_layout()

    plt.grid()

    plt.savefig(images_files_path + "/average_precision_vs_percentile.png")

    plt.close()

    y = list()

    for experiment in experiments[model_key]:
        y.append(len(experiment['selected_features_names']))

    plot4 = plt.plot()

    y_labels = []
    for key in accuracy_values:
        if key != 'total':
            y_labels.append(f'test on {key}')
            y = list()
            for x_value in x:
                y.append(stdev(accuracy_values[key][f'{x_value}']))
            plt.plot(x, y)

    y_labels = tuple(y_labels)
    plt.gca().legend(y_labels, loc='best', bbox_to_anchor=(1.05, 1))
    plt.tight_layout(pad=2)

    plt.title("stdev accuracy vs percentile")
    plt.xlabel(x_label)
    plt.ylabel("stdev accuracy")
    plt.grid()

    plt.savefig(images_files_path + "/stdev_accuracy_vs_percentile.png")

    plt.close()

    plot5 = plt.plot()

    y = list()
    for x_value in x:
        std_values = list()
        for key in accuracy_values:
            if key != 'total':
                std_values.append(stdev(accuracy_values[key][f'{x_value}']))
        y.append(mean(std_values))

    plt.plot(x, y)

    plt.title("average stdev accuracy vs percentile")
    plt.xlabel(x_label)
    plt.ylabel("stdev accuracy")
    plt.grid()

    plt.savefig(images_files_path + "/average_stdev_accuracy_vs_percentile.png")

    plt.close()

    plot6 = plt.plot()

    y = list()
    for x_value in x:
        std_values = list()
        for key in accuracy_values:
            if key == 'total':
                std_values.append(stdev(accuracy_values[key][f'{x_value}']))
        y.append(mean(std_values))

    plt.plot(x, y)

    plt.title("total stdev accuracy vs percentile")
    plt.xlabel(x_label)
    plt.ylabel("stdev accuracy")
    plt.grid()

    plt.savefig(images_files_path + "/total_stdev_accuracy_vs_percentile.png")

    plt.close()

    return None
##############################################################################################


def plot_time_resource_vs_selected_features(experiments, image_file_path, x_label):

    model_key = list(experiments.keys())[0]

    f = plt.figure(figsize=(6, 4))

    plt.title("time resources vs selected features")

    x = experiments['steps']

    # subplot(2, 1, 1)
    y = list()

    for experiment in experiments[model_key]:
        y.append(experiment['training_time'])


    plot1 = plt.subplot(2, 1, 1)
    plt.xlabel(x_label)
    plt.ylabel("training time (s)")
    plt.plot(x, y)
    plt.grid()


    # subplot(2, 1, 2)
    y = list()

    for experiment in experiments[model_key]:
        y.append(experiment['prediction_time'])
    factors = list()

    for element in y:
        factors.append(scientific_notation(element)[1])

    factor = min(factors)

    y = list(map(lambda x: x * (10 ** (abs(factor))), y))

    plot2 = plt.subplot(2, 1, 2)
    plt.xlabel(x_label)
    plt.ylabel(f"prediction time (e{factor} s)")
    plt.plot(x, y)
    plt.grid()

    plt.savefig(image_file_path + "/time_resource_vs_selected_features.png")

    plt.close()

    return None
##############################################################################################


def svm_from_params_dict(Xtrain, ytrain, Xtest, ytest, params_dict=None):

    if 'kernel' in params_dict:
        if params_dict['kernel'] == 'linear':
            model = svm.SVC(kernel=params_dict['kernel'], C=params_dict['C'], tol=params_dict['tol'],
                            shrinking=params_dict['shrinking'], verbose=True)

        elif params_dict['kernel'] == 'rbf':
            model = svm.SVC(kernel=params_dict['kernel'], gamma=params_dict['gamma'],  C=params_dict['C'],
                            tol=params_dict['tol'], shrinking=params_dict['shrinking'], verbose=True)

        elif params_dict['kernel'] == 'poly':
            model = svm.SVC(kernel=params_dict['kernel'], degree=params_dict['degree'], C=params_dict['C'],
                            gamma=params_dict['gamma'], coef0=params_dict['coef0'], tol=params_dict['tol'],
                            shrinking=params_dict['shrinking'], verbose=True)

    else:
        model = svm.LinearSVC(C=params_dict['C'], dual=params_dict['dual'], max_iter=params_dict['max_iter'],
                              penalty=params_dict['penalty'], loss=params_dict['loss'], tol=params_dict['tol'],
                              verbose=True)

    # Train Decision Tree Classifer
    start_training_time = time.time()
    model = model.fit(Xtrain, ytrain)
    stop_training_time = time.time()

    training_time = stop_training_time - start_training_time

    # Predict the response for test dataset
    start_prediction_time = time.time()
    predictions = model.predict(Xtest)
    stop_prediction_time = time.time()

    prediction_time = (stop_prediction_time - start_prediction_time) / len(Xtest)

    predictions = map(lambda x: x.round(0), predictions)
    predictions = list(predictions)

    report = classification_report(ytest, predictions, target_names=np.array(["NoEngagement", "Engagement"]),
                                   digits=4, output_dict=True)

    print()
    print(report)

    score = model.score(Xtest, ytest)
    print()
    print(f"accuracy = {round(score, 3)}")

    return {'model': model, 'accuracy': score, 'report': report, 'training_time': training_time,
            'prediction_time': prediction_time, 'predictions': predictions}
##############################################################################################


def tree_from_params_dict(Xtrain, Xtest, ytrain, ytest, params_dict=None, image_file_path=None):

    # Create Decision Tree classifer object
    model = DecisionTreeClassifier(criterion=params_dict['criterion'], max_depth=params_dict['max_depth'],
                                   splitter=params_dict['splitter'], min_samples_split=params_dict['min_samples_split'],
                                   min_samples_leaf=params_dict['min_samples_leaf'],
                                   min_impurity_decrease=params_dict['min_impurity_decrease'],
                                   max_features=params_dict['max_features'], ccp_alpha=params_dict['ccp_alpha'])

    # Train Decision Tree Classifer
    start_training_time = time.time()
    model = model.fit(Xtrain, ytrain)
    stop_training_time = time.time()

    training_time = stop_training_time - start_training_time

    # Predict the response for test dataset

    start_prediction_time = time.time()
    predictions = model.predict(Xtest)
    stop_prediction_time = time.time()

    prediction_time = (stop_prediction_time - start_prediction_time) / len(Xtest)

    predictions = map(lambda x: x.round(0), predictions)
    predictions = list(predictions)

    report = classification_report(ytest, predictions, target_names=np.array(["NoEngagement", "Engagement"]),
                                   digits=4, output_dict=True)

    print()
    print(report)
    score = metrics.accuracy_score(ytest, predictions)
    print(f"accuracy = {round(score, 3)}")

    # plot decision tree structure
    ####################################################################################################################
    """
    f_n = labels_generator.labels
    c_n = ['No engagement', 'Engagement']

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(40, 10), dpi=600)

    tree.plot_tree(model,
                   feature_names=f_n,
                   class_names=c_n,
                   filled=True)

    if image_file_path != None:
        fig.savefig(image_file_path)

    plt.close()
    ####################################################################################################################
    """
    return {'model': model, 'accuracy': score, 'report': report, 'training_time': training_time,
            'prediction_time': prediction_time, 'predictions': predictions}
################################################################################################


def knn_from_params_dict(Xtrain, Xtest, ytrain, ytest, params_dict=None):

    # Create Decision Tree classifer object
    model = KNeighborsClassifier(n_neighbors=params_dict['n_neighbors'], weights=params_dict['weights'],
                                 algorithm=params_dict['algorithm'], leaf_size=params_dict['leaf_size'],
                                 p=params_dict['p'], n_jobs=None)

    # Train Decision Tree Classifer
    start_training_time = time.time()
    model = model.fit(Xtrain, ytrain)
    stop_training_time = time.time()

    training_time = stop_training_time - start_training_time

    # Predict the response for test dataset

    start_prediction_time = time.time()
    predictions = model.predict(Xtest)
    stop_prediction_time = time.time()

    prediction_time = (stop_prediction_time - start_prediction_time) / len(Xtest)

    predictions = map(lambda x: x.round(0), predictions)
    predictions = list(predictions)

    report = classification_report(ytest, predictions, target_names=np.array(["NoEngagement", "Engagement"]),
                                   digits=4, output_dict=True)

    print()
    print(report)
    score = model.score(Xtest, ytest)
    print(f"accuracy = {round(score, 3)}")

    score = metrics.accuracy_score(ytest, predictions)

    return {'model': model, 'accuracy': score, 'report': report, 'training_time': training_time,
            'prediction_time': prediction_time, 'predictions': predictions}
################################################################################################


def random_forest_from_params_dict(Xtrain,Xtest,ytrain,ytest, params_dict=None):

    # Create Random Forest classifer object
    model = RandomForestClassifier(max_depth=params_dict['max_depth'], random_state=0, criterion=params_dict['criterion'],
                                   min_samples_split=params_dict['min_samples_split'],
                                   min_samples_leaf=params_dict['min_samples_leaf'],
                                   n_estimators=params_dict['n_estimators'], bootstrap=params_dict['bootstrap'],
                                   max_samples=params_dict['max_samples'], warm_start=params_dict['warm_start'],
                                   oob_score=params_dict['oob_score'], ccp_alpha=params_dict['ccp_alpha'], )

    # Train Decision Tree Classifer
    start_training_time = time.time()
    model = model.fit(Xtrain, ytrain)
    stop_training_time = time.time()

    training_time = stop_training_time - start_training_time

    # Predict the response for test dataset

    start_prediction_time = time.time()
    predictions = model.predict(Xtest)
    stop_prediction_time = time.time()

    prediction_time = (stop_prediction_time - start_prediction_time) / len(Xtest)

    predictions = map(lambda x: x.round(0), predictions)
    predictions = list(predictions)

    report = classification_report(ytest, predictions, target_names=np.array(["NoEngagement", "Engagement"]),
                                   digits=4, output_dict=True)
    print()
    print(report)
    score = model.score(Xtest, ytest)
    print(f"accuracy = {round(score, 3)}")

    score = metrics.accuracy_score(ytest, predictions)

    return {'model': model, 'accuracy': score, 'report': report, 'training_time': training_time,
            'prediction_time': prediction_time, 'predictions': predictions}
################################################################################################


def MLP_from_dict(Xtrain, Xval, Xtest, ytrain, yval, ytest, batch_size, epochs, callbacks, params_dict,
                  image_file_path=None, model_checkpoint_dir=None, best_models_path=None, verbose=1):

    # build the neural network
    ####################################################################################################################
    model = Sequential()

    for layer_index in range(params_dict['n_layers']):
        # input layer
        if layer_index == 0:
            model.add(Dense(params_dict['dense_values'][layer_index], params_dict['activation_functions'][layer_index],
                            input_shape=(Xtrain.shape[-1],)))
        # hidden layers
        else:
            model.add(Dense(params_dict['dense_values'][layer_index], params_dict['activation_functions'][layer_index]))

        model.add(Dropout(params_dict['drop_values'][layer_index]))

    # output layer
    if params_dict['activation_functions'][-1] == 'sigmoid':
        model.add(Dense(1, activation=params_dict['activation_functions'][-1]))
    if params_dict['activation_functions'][-1] == 'softmax':
        model.add(Dense(2, activation=params_dict['activation_functions'][-1]))
    ####################################################################################################################

    # compile
    ####################################################################################################################
    if params_dict['optimizer'] == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=params_dict['learning_rate'])

    elif params_dict['optimizer'] == "adamax":
        opt = tf.keras.optimizers.Adamax(learning_rate=params_dict['learning_rate'])

    if params_dict['activation_functions'][-1] == 'sigmoid':
        model.compile(loss='binary_crossentropy', optimizer=opt,
                      metrics=['accuracy'])

    if params_dict['activation_functions'][-1] == 'softmax':
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])
    ####################################################################################################################

    # show model parameters
    ####################################################################################################################
    print()
    print(model.summary())
    print()
    ####################################################################################################################

    # addestramento
    ####################################################################################################################
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{model_checkpoint_dir}/model_epoch_' + '{epoch:02d}.h5',
                                                          monitor="val_loss", verbose=0, save_best_only=False,
                                                          save_weights_only=False, mode="auto", save_freq="epoch",
                                                          options=None, initial_value_threshold=None)

    callbacks.extend([model_checkpoint])

    start_training_time = time.time()

    history = model.fit(Xtrain, ytrain, batch_size, epochs, verbose=verbose, validation_data=(Xval, yval),
                        shuffle=True, initial_epoch=0, callbacks=callbacks)

    stop_training_time = time.time()

    training_time = stop_training_time - start_training_time
    ####################################################################################################################

    ####################################################################################################################
    test_scores = {'test_loss': [], 'test_accuracy': []}
    models_names = os.listdir(model_checkpoint_dir)

    for i in range(len(models_names)):
        for model_name in models_names:
            if int(model_name.split("_")[-1].split(".")[0]) == i:
                model = keras.models.load_model(f'{model_checkpoint_dir}/{model_name}')
                score = model.evaluate(Xtest, ytest, verbose=0)
                test_scores['test_loss'].append(score[0])
                test_scores['test_accuracy'].append(score[1])
    ####################################################################################################################

    best_epoch = np.array(test_scores['test_accuracy']).argmax() + 1

    for model_name in models_names:
        if str(best_epoch) in model_name:
            best_model = keras.models.load_model(f"{model_checkpoint_dir}/model_epoch_{best_epoch:02d}.h5")

    if best_models_path:
        keras.models.save_model(filepath=best_models_path, model=best_model)

    # delete all checkpoints and save only the best model
    for model_name in models_names:
        os.remove(f"{model_checkpoint_dir}/{model_name}")

    # compute the time test
    ####################################################################################################################
    start_prediction_time = time.time()
    predictions = best_model.predict(Xtest, batch_size=batch_size)
    stop_prediction_time = time.time()

    prediction_time = (stop_prediction_time - start_prediction_time) / len(Xtest)
    ####################################################################################################################

    # give in output the results
    ####################################################################################################################

    score = best_model.evaluate(Xtest, ytest, verbose=verbose)

    if params_dict['activation_functions'][-1] == 'sigmoid':
        predictions = map(lambda x: x.round(0), predictions)
        predictions = list(predictions)

    if params_dict['activation_functions'][-1] == 'softmax':
        predictions = map(lambda x: x.argmax(), predictions)
        predictions = list(predictions)

    report = classification_report(ytest, predictions, target_names=np.array(["NoEngagement", "Engagement"]),
                                   digits=4, output_dict=True)

    # print report
    print()
    print(report)

    # plot results
    if image_file_path != None:
        NN_plot(history, test_scores, image_file_path)
    ####################################################################################################################

    return {'model': best_model, 'history': history, 'test_scores': test_scores, 'report': report, 'loss_and_accuracy': score,
            'training_time': training_time, 'prediction_time': prediction_time, 'predictions': predictions}
#############################################################################################


def features_selection_from_model(Xtrain, ytrain, Xval=[], Xtest=[], estimator=LogisticRegression(), threshold=None):

    selector = SelectFromModel(estimator=estimator, threshold=threshold).fit(Xtrain, ytrain)
    scores = selector.estimator_.coef_
    support = selector.get_support()
    selected_features_names = selector.get_feature_names_out(labels_generator.labels)
    newXtrain = selector.transform(Xtrain)
    newXval = []
    newXtest = []

    if len(Xval) > 0:
        newXval = selector.transform(Xval)

    if len(Xtest) > 0:
        newXtest = selector.transform(Xtest)

    return {'selector': selector, 'scores': scores, 'support': support,
            'transformed_Xtrain': newXtrain, 'transformed_Xval': newXval, 'transformed_Xtest': newXtest,
            'selected_features_names': selected_features_names}

def features_selection_from_score_function(Xtrain, ytrain, Xval=[], Xtest=[], percentile=10, function=f_classif, type='percentile'):

    if type == 'percentile':
        selector = SelectPercentile(function, percentile=percentile)

    elif type == 'Kbest':
        selector = SelectKBest(function, k=(int((percentile / 100) * len(Xtrain[0]))))

    selector.fit_transform(Xtrain, ytrain)
    scores = selector.scores_
    pvalues = selector.pvalues_
    newXtrain = selector.transform(Xtrain)
    # Mask feature names according to selected features.
    selected_features_names = selector.get_feature_names_out(labels_generator.labels)

    # Get parameters for this estimator.
    #selector.get_params()

    # Get a mask, or integer index, of the features selected.
    #selector.get_support() # Get a mask, or integer index, of the features selected.

    if len(Xval) > 0:
        newXval = selector.transform(Xval)
    else:
        newXval = Xval

    if len(Xtest) > 0:
        newXtest = selector.transform(Xtest)
    else:
        newXtest = Xtest

    return {'selector': selector, 'scores': scores, 'pvalues': pvalues, 'transformed_Xtrain': newXtrain,
            'transformed_Xval': newXval, 'transformed_Xtest': newXtest, 'selected_features_names': selected_features_names}
##############################################################################################


def features_selection_and_model_optimization(csv_file_path, test_subjects, validation_size, custom_steps=None,
                                              low_value=5, high_value=105, step=5,
                                              estimator=None, project_dir=None,
                                              project_name=None, feature_selection_type='from_score_function',
                                              model_type=None, groups=None, train_and_test_n_iteration=1, retest=False,
                                              class_labels=None):

    ####################################################################################################################
    try:
        os.mkdir(project_dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(project_dir + '/' + model_type)
    except FileExistsError:
        pass
    try:
        os.mkdir(project_dir + '/' + model_type + '/' + project_name)
    except FileExistsError:
        pass
    try:
        os.mkdir(project_dir + '/' + model_type + '/' + project_name + '/' + 'confusion_matrix_plots')
    except FileExistsError:
        pass
    ####################################################################################################################

    ####################################################################################################################
    model_files = os.listdir(f"features_selection_and_optimization/{model_type}/{project_name}")
    if 'experiments.json' in model_files:
        json_file = json.load(open(f"features_selection_and_optimization/{model_type}/{project_name}/experiments.json"))
        experiments = json_file[model_type]
        steps = json_file['steps']
    else:
        experiments = []
        steps = []

    if not custom_steps:
        range_ = list(np.arange(low_value, high_value, step))
    elif custom_steps:
        range_ = custom_steps
    ####################################################################################################################

    ####################################################################################################################
    dataset = get_dataset_from_csv(csv_file_path, [], 0.0)

    Xtrain = dataset['Xtrain']
    Xtest = dataset['Xval']
    Xtest_out = dataset['Xtest']
    ytrain = dataset['ytrain']
    ytest = dataset['yval']
    ytest_out = dataset['ytest']

    if feature_selection_type == 'from_model':
        selector = SelectFromModel(estimator=estimator, threshold=None).fit(Xtrain, ytrain)
        best_threshold = selector.threshold_
        range_.append(best_threshold)

    range_.sort()
    ####################################################################################################################

    for value in range_:

        if not retest and value in steps:
            break

        value = int(value)

        experiment = dict()

        total_y_pred = list()
        total_y_true = list()

        print('=' * 500 + '\n')
        if feature_selection_type == 'from_model':
            print(f"[INFO] testing with threshold = {round(value, 3)}")
        elif feature_selection_type == 'from_score_func':
            print(f"[INFO] testing with percentile = {round(value, 3)}")

        if feature_selection_type == 'from_model':
            experiment['threshold'] = value
            selected_features = features_selection_from_model(Xtrain, ytrain, Xval=Xtest, Xtest=Xtest_out, estimator=estimator,
                                                              threshold=value)

        elif feature_selection_type == 'from_score_func':
            experiment['percentile'] = value
            selected_features = features_selection_from_score_function(Xtrain, ytrain, Xval=Xtest, Xtest=Xtest_out,
                                                                       percentile=value, type='percentile')

        experiment['selected_features_names'] = list(selected_features['selected_features_names'])
        experiment['features_scores'] = list(selected_features['scores'])
        experiment['features_pvalues'] = list(selected_features['pvalues'])

        selector = selected_features['selector']

        transformed_Xtrain = selected_features['transformed_Xtrain']

        if len(dataset['Xval']) > 0:
            transformed_Xtest = selector.transform(dataset['Xval'])
        else:
            transformed_Xtest = np.array([])
        if len(dataset['Xtest']) > 0:
            transformed_Xtest_out = selector.transform(dataset['Xtest'])
        else:
            transformed_Xtest_out = np.array([])

        number_of_features = transformed_Xtrain.shape[1]

        experiment['number_of_features'] = number_of_features

        average_score = 0
        average_loss = 0

        experiment['report'] = list()

        for test_subject in test_subjects:

            try:
                os.mkdir(
                    project_dir + '/' + model_type + '/' + project_name + '/' + 'confusion_matrix_plots' + '/' +
                    f'on_test_{test_subject}')
            except FileExistsError:
                pass

            test_y_pred = list()
            test_y_true = list()

            ############################################################################################################
            dataset = get_dataset_from_csv(csv_file_path, [test_subject], validation_size)

            transformed_Xtrain = selector.transform(dataset['Xtrain'])
            transformed_ytrain = dataset['ytrain']

            if len(dataset['Xval']) > 0:
                transformed_Xtest = selector.transform(dataset['Xval'])
            else:
                transformed_Xtest = np.array([])

            transformed_ytest = dataset['yval']

            if len(dataset['Xtest']) > 0:
                transformed_Xtest_out = selector.transform(dataset['Xtest'])
            else:
                transformed_Xtest = np.array([])

            transformed_ytest_out = dataset['ytest']

            print('-' * 500 + '\n')
            print(f"[INFO] data_shape: train = {transformed_Xtrain.shape}, test = {transformed_Xtest.shape}, test out = {transformed_Xtest_out.shape}")
            print(f"[INFO] label_shape: train = {transformed_ytrain.shape}, test = {transformed_ytest.shape}test out = {transformed_ytest_out.shape}")
            print()
            ############################################################################################################

            if model_type.split("_")[0] == "SVM":

                ########################################################################################################
                if test_subjects.index(test_subject) == 0:

                    if value not in steps:

                        print('-' * 500 + '\n')
                        print()
                        print(f"[INFO] SVM model optimization procedure \n")

                        optimization_dataset = get_dataset_from_csv(csv_file_path, [], 0.0)

                        optimization_transformed_Xtrain = selector.transform(optimization_dataset['Xtrain'])
                        optimization_ytrain = optimization_dataset['ytrain']

                        optimization_result = sklearn_model_optimization(optimization_transformed_Xtrain, optimization_ytrain,
                                                                         model_type=model_type, groups=groups)
                    else:
                        optimization_result = dict()
                        for exp in experiments:
                            if ('percentile' in exp and exp['percentile'] == value) or \
                                    ('threshold' in exp and exp['threshold'] == value):
                                optimization_result['best_params'] = exp['best_params']
                                optimization_result['optimization_time'] = exp['optimization_time']

                    experiment['best_params'] = optimization_result['best_params']
                ########################################################################################################

                print('-' * 500 + '\n')
                print(f"[INFO] test on group {test_subject} \n")

                ########################################################################################################
                score = 0

                for train_and_test_index in range(train_and_test_n_iteration):

                    transformed_Xtrain, transformed_ytrain = shuffle(transformed_Xtrain, transformed_ytrain)

                    train_and_test_result = svm_from_params_dict(transformed_Xtrain, transformed_ytrain, transformed_Xtest_out, transformed_ytest_out,
                                                                 params_dict=optimization_result['best_params'])

                    score += train_and_test_result['accuracy']
                    experiment['report'].append({'group': test_subject, 'report': train_and_test_result['report']})

                    test_y_pred.extend(train_and_test_result['predictions'])
                    test_y_true.extend(transformed_ytest_out)

                confusion_matrix = ConfusionMatrixDisplay.from_predictions(test_y_true, test_y_pred, display_labels=class_labels)

                plt.tight_layout(pad=2)

                plt.savefig(project_dir + '/' + model_type + '/' + project_name + '/' + 'confusion_matrix_plots' + '/' +
                            f'on_test_{test_subject}' + '/' + f'confusion_matrix_on_value_{value}.png')

                plt.close()

                score = score / train_and_test_n_iteration
                ########################################################################################################

            elif model_type == "DECISION_TREE":

                ########################################################################################################
                if test_subjects.index(test_subject) == 0:

                    if value not in steps:

                        print('-' * 500 + '\n')
                        print(f"[INFO] DECISION TREE model optimization procedure \n")

                        optimization_dataset = get_dataset_from_csv(csv_file_path, [], 0.0)

                        optimization_transformed_Xtrain = selector.transform(optimization_dataset['Xtrain'])
                        optimization_ytrain = optimization_dataset['ytrain']

                        optimization_result = sklearn_model_optimization(optimization_transformed_Xtrain, optimization_ytrain,
                                                                         model_type=model_type, groups=groups)
                    else:
                        optimization_result = dict()
                        for exp in experiments:
                            if ('percentile' in exp and exp['percentile'] == value) or \
                                    ('threshold' in exp and exp['threshold'] == value):
                                optimization_result['best_params'] = exp['best_params']
                                optimization_result['optimization_time'] = exp['optimization_time']

                    experiment['best_params'] = optimization_result['best_params']
                ########################################################################################################

                print('-' * 500 + '\n')
                print(f"[INFO] test on group {test_subject} \n")

                ########################################################################################################

                try:
                    os.mkdir(
                        project_dir + '/' + model_type + '/' + project_name + '/' + 'models_representation')
                except FileExistsError:
                    pass
                try:
                    os.mkdir(
                        project_dir + '/' + model_type + '/' + project_name + '/' + 'models_representation' + '/' +
                        f'on_test_{test_subject}')
                except FileExistsError:
                    pass
                try:
                    os.mkdir(
                        project_dir + '/' + model_type + '/' + project_name + '/' + 'models_representation' + '/' +
                        f'on_test_{test_subject}/on_value_{value}')
                except FileExistsError:
                    pass

                score = 0

                for train_and_test_index in range(train_and_test_n_iteration):

                    transformed_Xtrain, transformed_ytrain = shuffle(transformed_Xtrain, transformed_ytrain)

                    train_and_test_result = tree_from_params_dict(transformed_Xtrain,  transformed_Xtest_out, transformed_ytrain, transformed_ytest_out,
                                                                  params_dict=optimization_result['best_params'],
                                                                  image_file_path=f"{project_dir}/{model_type}/{project_name}/"
                                                                                  f"models_representation/on_test_{test_subject}/"
                                                                                  f"on_value_{value}/model_representation_iter_"
                                                                                  f"{train_and_test_index}.png")

                    score += train_and_test_result['accuracy']
                    experiment['report'].append({'group': test_subject, 'report': train_and_test_result['report']})

                    test_y_pred.extend(train_and_test_result['predictions'])
                    test_y_true.extend(transformed_ytest_out)

                confusion_matrix = ConfusionMatrixDisplay.from_predictions(test_y_true, test_y_pred, display_labels=class_labels)

                plt.savefig(project_dir + '/' + model_type + '/' + project_name + '/' + 'confusion_matrix_plots' + '/' +
                            f'on_test_{test_subject}' + '/' + f'confusion_matrix_on_value_{value}.png')

                plt.tight_layout(pad=2)

                plt.close()

                score = score / train_and_test_n_iteration
                ########################################################################################################

            elif model_type == "RANDOM_FOREST":

                ########################################################################################################
                if test_subjects.index(test_subject) == 0:

                    if value not in steps:

                        print('-' * 500 + '\n')
                        print(f"[INFO] RANDOM FOREST model optimization procedure \n")

                        optimization_dataset = get_dataset_from_csv(csv_file_path, [], 0.0)

                        optimization_transformed_Xtrain = selector.transform(optimization_dataset['Xtrain'])
                        optimization_ytrain = optimization_dataset['ytrain']

                        optimization_result = sklearn_model_optimization(optimization_transformed_Xtrain, optimization_ytrain,
                                                                         model_type=model_type, groups=groups)
                    else:
                        optimization_result = dict()
                        for exp in experiments:
                            if ('percentile' in exp and exp['percentile'] == value) or \
                                    ('threshold' in exp and exp['threshold'] == value):
                                optimization_result['best_params'] = exp['best_params']
                                optimization_result['optimization_time'] = exp['optimization_time']

                    experiment['best_params'] = optimization_result['best_params']
                ########################################################################################################

                print('-' * 500 + '\n')
                print(f"[INFO] test on group {test_subject} \n")

                ########################################################################################################

                score = 0

                for train_and_test_index in range(train_and_test_n_iteration):

                    transformed_Xtrain, transformed_ytrain = shuffle(transformed_Xtrain, transformed_ytrain)

                    train_and_test_result = random_forest_from_params_dict(transformed_Xtrain, transformed_Xtest_out,
                                                                           transformed_ytrain, transformed_ytest_out,
                                                                           params_dict=optimization_result['best_params'])

                    score += train_and_test_result['accuracy']
                    experiment['report'].append({'group': test_subject, 'report': train_and_test_result['report']})

                    test_y_pred.extend(train_and_test_result['predictions'])
                    test_y_true.extend(transformed_ytest_out)

                confusion_matrix = ConfusionMatrixDisplay.from_predictions(test_y_true, test_y_pred, display_labels=class_labels)

                plt.savefig(project_dir + '/' + model_type + '/' + project_name + '/' + 'confusion_matrix_plots' + '/' +
                            f'on_test_{test_subject}' + '/' + f'confusion_matrix_on_value_{value}.png')

                plt.tight_layout(pad=2)

                plt.close()

                score = score / train_and_test_n_iteration
                ########################################################################################################

            elif model_type == "KNN":

                ########################################################################################################
                if test_subjects.index(test_subject) == 0:

                    if value not in steps:

                        print('-' * 500 + '\n')
                        print(f"[INFO] KNN model optimization procedure \n")

                        optimization_dataset = get_dataset_from_csv(csv_file_path, [], 0.0)

                        optimization_transformed_Xtrain = selector.transform(optimization_dataset['Xtrain'])
                        optimization_ytrain = optimization_dataset['ytrain']

                        optimization_result = sklearn_model_optimization(optimization_transformed_Xtrain, optimization_ytrain,
                                                                         model_type=model_type, groups=groups)
                    else:
                        optimization_result = dict()
                        for exp in experiments:
                            if ('percentile' in exp and exp['percentile'] == value) or \
                                    ('threshold' in exp and exp['threshold'] == value):
                                optimization_result['best_params'] = exp['best_params']
                                optimization_result['optimization_time'] = exp['optimization_time']

                    experiment['best_params'] = optimization_result['best_params']
                ########################################################################################################

                print('-' * 500 + '\n')
                print(f"[INFO] test on group {test_subject} \n")

                ########################################################################################################
                score = 0

                for train_and_test_index in range(train_and_test_n_iteration):

                    transformed_Xtrain, transformed_ytrain = shuffle(transformed_Xtrain, transformed_ytrain)

                    train_and_test_result = knn_from_params_dict(transformed_Xtrain, transformed_Xtest_out,
                                                                 transformed_ytrain, transformed_ytest_out,
                                                                 params_dict=optimization_result['best_params'])

                    score += train_and_test_result['accuracy']
                    experiment['report'].append({'group': test_subject, 'report': train_and_test_result['report']})

                    test_y_pred.extend(train_and_test_result['predictions'])
                    test_y_true.extend(transformed_ytest_out)

                confusion_matrix = ConfusionMatrixDisplay.from_predictions(test_y_true, test_y_pred, display_labels=class_labels)

                plt.savefig(project_dir + '/' + model_type + '/' + project_name + '/' + 'confusion_matrix_plots' + '/' +
                            f'on_test_{test_subject}' + '/' + f'confusion_matrix_on_value_{value}.png')

                plt.tight_layout(pad=2)

                plt.close()

                score = score / train_and_test_n_iteration
                ########################################################################################################

            elif 'MLP' in model_type:

                ########################################################################################################
                if test_subjects.index(test_subject) == 0:

                    if value not in steps:

                        print('-' * 500 + '\n')
                        print(f"[INFO] MLP model optimization procedure \n")

                        model_builders = [model_builder]
                        model_names = [model_type]

                        optimization_dataset = get_dataset_from_csv(csv_file_path, [], validation_size)

                        optimization_transformed_Xtrain = selector.transform(optimization_dataset['Xtrain'])
                        optimization_ytrain = optimization_dataset['ytrain']
                        optimization_transformed_Xtest = selector.transform(optimization_dataset['Xval'])
                        optimization_ytest = optimization_dataset['yval']
                        optimization_transformed_Xtest_out = optimization_dataset['Xtest']
                        optimization_ytest_out = optimization_dataset['ytest']


                        optimization_result = keras_model_optimization(optimization_transformed_Xtrain, optimization_ytrain,
                                                                       optimization_transformed_Xtest, optimization_ytest,
                                                                       optimization_transformed_Xtest_out, optimization_ytest_out,
                                                                       optimization_transformed_Xtrain[0].shape,
                                                                       model_builders, model_names)
                    else:
                        optimization_result = dict()
                        for exp in experiments:
                            if ('percentile' in exp and exp['percentile'] == value) or \
                                    ('threshold' in exp and exp['threshold'] == value):
                                optimization_result['best_params'] = exp['best_params']
                                optimization_result['optimization_time'] = exp['optimization_time']
                                optimization_result['batch_size'] = exp['batch_size']

                    experiment['best_params'] = optimization_result['best_params']
                    experiment['batch_size'] = optimization_result['batch_size']
                ########################################################################################################

                print('-' * 500 + '\n')
                print(f"[INFO] test on group {test_subject} \n")

                ########################################################################################################
                stop_early_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
                stop_early_accuracy = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

                callbacks = [stop_early_loss, stop_early_accuracy]

                score = [0, 0]

                try:
                    os.mkdir(
                        project_dir + '/' + model_type + '/' + project_name + '/' + 'plots_accuracy_and_loss')
                except FileExistsError:
                    pass
                try:
                    os.mkdir(
                        project_dir + '/' + model_type + '/' + project_name + '/' + 'plots_accuracy_and_loss' + '/' +
                        f'on_test_{test_subject}')
                except FileExistsError:
                    pass
                try:
                    os.mkdir(
                        project_dir + '/' + model_type + '/' + project_name + '/' + 'plots_accuracy_and_loss' + '/' +
                        f'on_test_{test_subject}/on_value_{value}')
                except FileExistsError:
                    pass
                try:
                    os.mkdir(
                        project_dir + '/' + model_type + '/' + project_name + '/' + 'model_checkpoints')
                except FileExistsError:
                    pass
                try:
                    os.mkdir(
                        project_dir + '/' + model_type + '/' + project_name + '/' + 'model_checkpoints' + '/' +
                        f'on_test_{test_subject}')
                except FileExistsError:
                    pass
                try:
                    os.mkdir(
                        project_dir + '/' + model_type + '/' + project_name + '/' + 'model_checkpoints' + '/' +
                        f'on_test_{test_subject}/on_value_{value}')
                except FileExistsError:
                    pass
                try:
                    os.mkdir(
                        project_dir + '/' + model_type + '/' + project_name + '/' + 'best_models')
                except FileExistsError:
                    pass
                try:
                    os.mkdir(
                        project_dir + '/' + model_type + '/' + project_name + '/' + 'best_models' + '/' +
                        f'on_test_{test_subject}')
                except FileExistsError:
                    pass
                try:
                    os.mkdir(
                        project_dir + '/' + model_type + '/' + project_name + '/' + 'best_models' + '/' +
                        f'on_test_{test_subject}/on_value_{value}')
                except FileExistsError:
                    pass

                for train_and_test_index in range(train_and_test_n_iteration):

                    transformed_Xtrain, transformed_ytrain = shuffle(transformed_Xtrain, transformed_ytrain)
                    transformed_Xtest, transformed_ytest = shuffle(transformed_Xtest, transformed_ytest)

                    train_and_test_result = MLP_from_dict(transformed_Xtrain, transformed_Xtest, transformed_Xtest_out,
                                                          transformed_ytrain, transformed_ytest, transformed_ytest_out,
                                                          batch_size=optimization_result['batch_size'],
                                                          epochs=200,
                                                          callbacks=callbacks,
                                                          params_dict=optimization_result['best_params'],
                                                          verbose=1,
                                                          image_file_path=f"{project_dir}/{model_type}/{project_name}/"
                                                                          f"plots_accuracy_and_loss/on_test_{test_subject}/"
                                                                          f"on_value_{value}/plot_iter_"
                                                                          f"{train_and_test_index}.png",
                                                          model_checkpoint_dir=f"{project_dir}/{model_type}/{project_name}/"
                                                                          f"model_checkpoints/on_test_{test_subject}/"
                                                                               f"on_value_{value}",
                                                          best_models_path=f"{project_dir}/{model_type}/{project_name}/"
                                                                          f"best_models/on_test_{test_subject}/"
                                                                          f"on_value_{value}/best_model_iter_"
                                                                           f"{train_and_test_index}.h5")

                    score[0] += train_and_test_result['loss_and_accuracy'][0]
                    score[1] += train_and_test_result['loss_and_accuracy'][1]
                    experiment['report'].append({'group': test_subject, 'report': train_and_test_result['report']})

                    test_y_pred.extend(train_and_test_result['predictions'])
                    test_y_true.extend(transformed_ytest_out)

                confusion_matrix = ConfusionMatrixDisplay.from_predictions(test_y_true, test_y_pred, display_labels=class_labels)

                plt.savefig(project_dir + '/' + model_type + '/' + project_name + '/' + 'confusion_matrix_plots' + '/' +
                            f'on_test_{test_subject}' + '/' + f'confusion_matrix_on_value_{value}.png')

                plt.tight_layout(pad=2)

                plt.close()

                score[0] = score[0] / train_and_test_n_iteration
                score[1] = score[1] / train_and_test_n_iteration
                ########################################################################################################

            total_y_pred.extend(test_y_pred)
            total_y_true.extend(test_y_true)

            if 'MLP' in model_type:
                average_score += score[1]
                average_loss += score[0]
            else:
                average_score += score

        ################################################################################################################
        try:
            os.mkdir(
                project_dir + '/' + model_type + '/' + project_name + '/' + 'confusion_matrix_plots' + '/' + 'on_total')
        except FileExistsError:
            pass

        confusion_matrix = ConfusionMatrixDisplay.from_predictions(total_y_true, total_y_pred,
                                                                   display_labels=class_labels)

        plt.tight_layout(pad=2)

        plt.savefig(project_dir + '/' + model_type + '/' + project_name + '/' + 'confusion_matrix_plots' + '/' +
                    f'on_total' + '/' + f'confusion_matrix_on_value_{value}.png')

        plt.close()
        ################################################################################################################

        experiment['optimization_time'] = optimization_result['optimization_time']
        experiment['training_time'] = train_and_test_result['training_time']
        experiment['prediction_time'] = train_and_test_result['prediction_time']


        average_score = average_score / len(test_subjects)
        average_loss = average_loss / len(test_subjects)

        experiment['test_accuracy'] = average_score
        if 'MLP' in model_type:
            experiment['test_loss'] = average_loss

        exp_index = None
        for exp in experiments:
            if 'percentile' in exp:
                if exp['percentile'] == value:
                    exp_index = experiments.index(exp)
                    experiments.remove(exp)

            elif 'threshold' in exp:
                if exp['threshold'] == value:
                    exp_index = experiments.index(exp)
                    experiments.remove(exp)

        if exp_index!=None:
            experiments.insert(exp_index, experiment)
        else:
            experiments.append(experiment)

        if value not in steps:
            steps.append(value)
            steps.sort()

        """
        ################################################################################################################
        data = {f'{model_type}': experiments, 'steps': steps}

        with open(f"{project_dir}/{model_type}/{project_name}/experiments.json", "w") as outfile:
            json.dump(data, outfile)
        ################################################################################################################
        
        ################################################################################################################
        if project_dir and feature_selection_type == 'from_model':
            plot_metrics_vs_selected_features(data, project_dir + '/' + model_type + '/' + project_name,
                                              x_label='threshold', class_labels=class_labels)
            plot_time_resource_vs_selected_features(data, project_dir + '/' + model_type + '/' + project_name,
                                                    x_label='percentile')
        elif project_dir and feature_selection_type == 'from_score_func':
            plot_metrics_vs_selected_features(data, project_dir + '/' + model_type + '/' + project_name,
                                              x_label='percentile', class_labels=class_labels)
            plot_time_resource_vs_selected_features(data, project_dir + '/' + model_type + '/' + project_name,
                                                    x_label='percentile')
        ################################################################################################################
        """
    ####################################################################################################################
    data = {f'{model_type}': experiments, 'steps': steps}
    with open(f"{project_dir}/{model_type}/{project_name}/experiments.json", "w") as outfile:
        json.dump(data, outfile)
    ####################################################################################################################

    ####################################################################################################################
    if project_dir and feature_selection_type == 'from_model':
        plot_metrics_vs_selected_features(data, project_dir + '/' + model_type + '/' + project_name,
                                          x_label='threshold', class_labels=class_labels)
        plot_time_resource_vs_selected_features(data, project_dir + '/' + model_type + '/' + project_name,
                                                x_label='percentile')
    elif project_dir and feature_selection_type == 'from_score_func':
        plot_metrics_vs_selected_features(data, project_dir + '/' + model_type + '/' + project_name,
                                          x_label='percentile', class_labels=class_labels)
        plot_time_resource_vs_selected_features(data, project_dir + '/' + model_type + '/' + project_name,
                                                x_label='percentile')
    ####################################################################################################################

    return experiments



















































































































