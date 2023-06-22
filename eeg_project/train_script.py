######################################################################################
######################################################################################
import os

from function import *
from predictive_models import *
import joblib
import random
######################################################################################
######################################################################################
test_subjects = [i for i in range(3)]
n_iterations = 5

with open("dataset_norm/dataset_norm.csv", newline="", encoding="ISO-8859-1") as filecsv:
    lettore = csv.reader(filecsv, delimiter=",")
    dataset = []
    for row in lettore:
        dataset.append(row)
for row in dataset:
    for i in range(len(row)):
        row[i] = float(row[i])
##############################CREAZIONE TRAIN E TEST#########################################
# scompongo il dataset in train(70%) e test(30%)
# separo i target dal dataset (primo elemento di ogni riga)
# trasformo il campione in un array con numpy
# trasformo ogni elemento in un vettore di lunghezza pari al numero di caratteristiche con numpy
# dunque trasformo il dataset (campione di elementi) in un array 2-D (matrice)
#############################################################################################
for test_index in test_subjects:

    for iteration_index in range(n_iterations):

        split_dataset = split_by_subjects(dataset, test_subjects=[test_index], validation_size=0.3)

        Xtrain = split_dataset["Xtrain"]
        Xvalidation = split_dataset["Xvalidation"]
        Xtest = split_dataset["Xtest"]

        ytrain = []

        for element in Xtrain:
            ytrain.append(element[-1])
        for i in range(len(Xtrain)):
            Xtrain[i] = Xtrain[i][:len(Xtrain[i])-1]

        yvalidation = []
        for elemento in Xvalidation:
            yvalidation.append(int(elemento[-1]))
        for i in range(len(Xvalidation)):
            Xvalidation[i] = Xvalidation[i][:len(Xvalidation[i])-1]

        ytest = []
        for elemento in Xtest:
            ytest.append(int(elemento[-1]))
        for i in range(len(Xtest)):
            Xtest[i] = Xtest[i][:len(Xtest[i])-1]

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

        print(f"[INFO] data_shape: train = {Xtrain.shape}, validation = {Xvalidation.shape}, test = {Xtest.shape}")
        print(f"[INFO] label_shape: train = {ytrain.shape}, validation = {yvalidation.shape}, test = {ytest.shape}")
        print()

        ##############################################################################################
        #################################SUPPORT VECTOR MACHINE#######################################

        # construct four type of 'support vector machine'

        print("#"*53)
        print("#" * 25 + "SVC" + "#" * 25)
        print()

        model_name = "svc"

        try:
            os.mkdir(f"models/{model_name}")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"models/{model_name}/on-test-{test_index}")
        except FileExistsError:
            pass

        score1, svm = SVM(Xtrain, ytrain, Xtest, ytest, type='svc')

        joblib.dump(svm, f"models/{model_name}/on-test-{test_index}/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.sav")

        print()
        print("#"*57)
        print("#" * 25 + "RBF_SVC" + "#" * 25)
        print()

        model_name = "rbf-svc"

        try:
            os.mkdir(f"models/{model_name}")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"models/{model_name}/on-test-{test_index}")
        except FileExistsError:
            pass

        score, svm = SVM(Xtrain, ytrain, Xtest, ytest, type='rbf_svc')

        joblib.dump(svm, f"models/{model_name}/on-test-{test_index}/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.sav")

        print()
        print("#"*58)
        print("#" * 25 + "POLY_SVC" + "#" * 25)
        print()

        model_name = "poly-svc"

        try:
            os.mkdir(f"models/{model_name}")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"models/{model_name}/on-test-{test_index}")
        except FileExistsError:
            pass

        score, svm = SVM(Xtrain, ytrain, Xtest, ytest, type='poly_svc')

        joblib.dump(svm, f"models/{model_name}/on-test-{test_index}/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.sav")

        print()
        print("#"*58)
        print("#" * 25 + "LIN_SVC" + "#" * 25)
        print()

        model_name = "lin-svc"

        try:
            os.mkdir(f"models/{model_name}")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"models/{model_name}/on-test-{test_index}")
        except FileExistsError:
            pass

        score, svm = SVM(Xtrain, ytrain, Xtest, ytest, type='lin_svc')

        joblib.dump(svm, f"models/{model_name}/on-test-{test_index}/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.sav")

        print()

        ##############################################################################################
        ######################################DECISION TREE###########################################

        # construct different type of 'decision tree'

        print()
        print("#" * 63)
        print("#" * 25 + f"TREE CLASSIFIER" + "#" * 25)
        print()

        model_name ="tree"

        try:
            os.mkdir(f"models/{model_name}")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"models/{model_name}/on-test-{test_index}")
        except FileExistsError:
            pass

        score, tree = TREE(Xtrain, Xtest, ytrain, ytest, criterion="entropy", maxdepth=3,
                           image_file_path=f"image_decision_tree/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.png")

        joblib.dump(tree, f"models/{model_name}/on-test-{test_index}/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.sav")

        # construct different type of 'random forest'

        print()
        print("#" * 63)
        print("#" * 25 + f"RANDOM FOREST CLASSIFIER" + "#" * 25)
        print()

        model_name = "random-forest"

        try:
            os.mkdir(f"models/{model_name}")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"models/{model_name}/on-test-{test_index}")
        except FileExistsError:
            pass

        score, random_forest = RANDOM_FOREST(Xtrain, Xtest, ytrain, ytest, criterion="entropy", maxdepth=3, n_estimators=100)

        joblib.dump(random_forest, f"models/{model_name}/on-test-{test_index}/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.sav")

        ######################################NEURAL NETWORK #########################################

        # construct two type of 'Neural Network' with only one hidden layer

        print()
        print("#" * 88)
        print("#" * 25 + "NN(ACTIVATION_OUTPUT=SIGMOID OPT=ADAM)" + "#" * 25)
        print()

        model_name = "nn_layers_units_drop-out_output-activation_optimizers_batch-size_lr_epochs"

        try:
            os.mkdir(f"models/{model_name}")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"models/{model_name}/on-test-{test_index}")
        except FileExistsError:
            pass
        """
        history, report, score, nn = NN_dense_drop(Xtrain, Xvalidation, Xtest, ytrain, yvalidation, ytest, batch_size=8, epochs=1,
                                                   n_dense=736, drop_value=0.2 , activation=('relu', 'sigmoid'),
                                                   optimizer='adam', learning_rate=0.001,
                                                   image_file_path=f"plots_NN/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.png",
                                                   model_file_path=f"models/{model_name}/on-test-{test_index}/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.h5",
                                                   fit_on_best_epoch=True)

        print()
        print("#" * 88)
        print("#" * 25 + "NN(ACTIVATION_OUTPUT=SOFTMAX OPT=ADAM)" + "#" * 25)
        print()

        model_name = "nn_layers_units_drop-out_output-activation_optimizers_batch-size_lr_epochs"

        try:
            os.mkdir(f"models/{model_name}")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"models/{model_name}/on-test-{test_index}")
        except FileExistsError:
            pass

        history, report, score, nn = NN_dense_drop(Xtrain, Xvalidation, Xtest, ytrain, yvalidation, ytest, batch_size=8, epochs=1,
                                                   n_dense=232, drop_value=0.2, activation=('relu', 'softmax'),
                                                   optimizer='adam', learning_rate=0.001,
                                                   image_file_path=f"plots_NN/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.png",
                                                   model_file_path=f"models/{model_name}/on-test-{test_index}/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.h5",
                                                   fit_on_best_epoch=True)

        print()
        print("#" * 90)
        print("#" * 25 + "NN(ACTIVATION_OUTPUT=SIGMOID OPT=ADAMAX)" + "#" * 25)
        print()

        model_name = "nn_layers_units_drop-out_output-activation_optimizers_batch-size_lr_epochs"

        try:
            os.mkdir(f"models/{model_name}")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"models/{model_name}/on-test-{test_index}")
        except FileExistsError:
            pass

        history, report, score, nn = NN_dense_drop(Xtrain, Xvalidation, Xtest, ytrain, yvalidation, ytest, batch_size=8, epochs=1,
                                                   n_dense=736, drop_value=0.2, activation=('relu', 'sigmoid'),
                                                   optimizer='adamax', learning_rate=0.001,
                                                   image_file_path=f"plots_NN/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.png",
                                                   model_file_path=f"models/{model_name}/on-test-{test_index}/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.h5",
                                                   fit_on_best_epoch=True)

        print()
        print("#" * 90)
        print("#" * 25 + "NN(ACTIVATION_OUTPUT=SOFTMAX OPT=ADAMAX)" + "#" * 25)
        print()

        model_name = "nn_layers_units_drop-out_output-activation_optimizers_batch-size_lr_epochs"

        try:
            os.mkdir(f"models/{model_name}")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"models/{model_name}/on-test-{test_index}")
        except FileExistsError:
            pass

        history, report, score, nn = NN_dense_drop(Xtrain, Xvalidation, Xtest, ytrain, yvalidation, ytest, batch_size=8, epochs=1,
                                                   n_dense=720, drop_value=0.2, activation=('relu', 'softmax'),
                                                   optimizer='adamax', learning_rate=0.001,
                                                   image_file_path=f"plots_NN/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.png",
                                                   model_file_path=f"models/{model_name}/on-test-{test_index}/{model_name}_on-test-{test_index}_iteration-{iteration_index}_.h5",
                                                   fit_on_best_epoch=True)

        print()
        print("#" * 90)
        """
        ##############################################################################################
        ##############################################################################################




