import os
import keras
import joblib
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from function import *


def test(model_path, Xtest, ytest, labels, model_type="sav", output_type="1"):

    model_name = model_path.split("/")[-1]

    if model_type == "h5":
        model = keras.models.load_model(model_path)
    elif model_type == "sav":
        model = joblib.load(model_path)

    predictions = model.predict(Xtest)

    if output_type == "1":
        predictions = map(lambda x: x.round(0), predictions)
    elif output_type == "2":
        predictions = map(lambda x: x.argmax(), predictions)

    predictions = list(predictions)

    metrics_report = classification_report(ytest, predictions,
                                           target_names=labels,
                                           digits=4,
                                           output_dict=True)

    confusion_matrix = ConfusionMatrixDisplay.from_predictions(ytest, predictions, display_labels=labels)

    plt.savefig(f"confusion_matrix_plots/confusion_matrix_{model_name}.png")

    return metrics_report


def multi_test(models_dir, Xtest, ytest, labels, test_index, model_type="sav", output_type="1"):

    model_name = models_dir.split("/")[-1]
    models_names = os.listdir(models_dir)

    models = []

    if model_type == "h5":
        for model_name in models_names:
            model_path = f"{models_dir}/{model_name}"
            models.append(keras.models.load_model(model_path))
    elif model_type == "sav":
        for model_name in models_names:
            model_path = f"{models_dir}/{model_name}"
            models.append(joblib.load(model_path))

    predictions = multi_prediction(Xtest, models, output_type=output_type)

    predictions = list(predictions)

    metrics_report = classification_report(ytest, predictions,
                                           target_names=labels,
                                           digits=4,
                                           output_dict=True)

    confusion_matrix = ConfusionMatrixDisplay.from_predictions(ytest, predictions, display_labels=labels)

    plt.savefig(f"confusion_matrix_plots/confusion_matrix_{model_name}.png")

    return metrics_report


def multi_prediction(Xtest, models, output_type="1"):

    pred = np.zeros(len(Xtest))

    for model in models:

        ######################################################################################
        ######################################################################################

        predictions = model.predict(Xtest)

        if output_type == '1':
            #predictions = map(lambda x: x.round(0), predictions)
            predictions = list(predictions)
            pred += np.array(predictions)

        elif output_type == '2':
            predictions = map(lambda x: x.argmax(), predictions)
            predictions = list(predictions)
            pred += np.array(predictions)

    pred = pred / len(models)

    #if output_type == 1:
    #    for value in pred:
    #        print(value)

    pred = map(lambda x: x.round(0), pred)
    pred = list(pred)

    return pred


if __name__ == "__main__":

    from predictive_models import *

    with open("dataset_norm/dataset_norm.csv", newline="", encoding="ISO-8859-1") as filecsv:
        lettore = csv.reader(filecsv, delimiter=",")
        dataset = []
        for row in lettore:
            dataset.append(row)
    for row in dataset:
        for i in range(len(row)):
            row[i] = float(row[i])

    models_directories = os.listdir("models")

    for base_model in models_directories:

        test_index_values = os.listdir(f"models/{base_model}")

        metrics_dicts = list()

        for test_index in test_index_values:

            test_index = int(test_index.split("-")[-1])

            split_dataset = split_by_subjects(dataset, test_subjects=[test_index], validation_size=0.3)

            Xtest = split_dataset["Xtest"]

            ytrain = []

            ytest = []
            for element in Xtest:
                ytest.append(int(element[-1]))
            for i in range(len(Xtest)):
                Xtest[i] = Xtest[i][:len(Xtest[i]) - 1]

            Xtest = np.array(Xtest)

            ytest = np.array(ytest)

            # set dimension
            height_test = len(Xtest)
            width = len(Xtest[0])
            channels = 1

            Xtest = Xtest.reshape(height_test, width, )

            models_dir = f"models/{base_model}/on-test-{test_index}"

            models = os.listdir(models_dir)
            labels = list()


            #model_path = f"{model_dir}/{model_name}"

            model_type = models[0].split(".")[-1]

            if "softmax" in models[0]:
                output_type = "2"
            else:
                output_type = "1"

            result = multi_test(models_dir, Xtest, ytest, labels=np.array(["NoEngagement", "Engagement"]),
                                model_type=model_type, output_type=output_type, test_index=test_index)

            metrics_dict = {}
            for key in result:
                if str(type(result[key])) == "<class 'dict'>":
                    for sub_key in result[key]:
                        print(f"{key}_{sub_key} = {result[key][sub_key]}")
                        metrics_dict[f"{key}_{sub_key}"] = result[key][sub_key]
                else:
                    print(f"{key} = {result[key]}")
                    metrics_dict[f"{key}"] = result[key]
            metrics_dict["test index"] = test_index
            metrics_dicts.append(metrics_dict)

            for key in metrics_dicts[0]:
                labels.append(key)
            labels.append("test index")
            # salva file .json contenente le metriche per singolo test index
            from_dicts_to_csv([metrics_dict], labels, f"test_results/{base_model}_on-test-{test_index}_results.csv")

        from_dicts_to_csv(metrics_dicts, labels, f"test_results/{base_model}_results.csv")