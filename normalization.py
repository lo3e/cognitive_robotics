import csv
from numpy import mean, std
import os
from function import *
#######################################################################################
#######################################################################################


def smussa(data, soglia=0.95):
    value=0

    while True:
        value += 1
        conteiner = []

        for element in data:

            if element < value:
                conteiner.append(element)

        if len(conteiner)/len(data) > soglia:

            for i in range(len(data)):

                if data[i] > value:
                    data[i] = value

            return data
#######################################################################################
#######################################################################################


def unioneEnormalizzazione(lista_file, directory):

    dataset=[]
    for file in lista_file:

        with open(f"{directory}/"+str(file), newline="", encoding="ISO-8859-1") as filecsv:
            lettore = csv.reader(filecsv, delimiter=",")
            dataset_ = []

            for row in lettore:
                dataset_.append(row)

        for row in dataset_:

            for i in range(len(row)):
                row[i] = float(row[i])

        ##########################################################################
        ##########################################################################

        for i in range(len(dataset_[0]) - 1):
            lista_temp = []

            for j in range(len(dataset_)):
                lista_temp.append(dataset_[j][i])

            lista_temp = smussa(lista_temp, 0.95)

            for k in range(len(dataset_)):
                dataset_[k][i] = lista_temp[k]

        lista_ = [[], [], [], []]  # mean,max,min,std
        for i in range(len(dataset_[0]) - 1):
            lista_temp = []
            for j in range(len(dataset_)):
                lista_temp.append(dataset_[j][i])
            lista_stat_temp = [mean(lista_temp), max(lista_temp), min(lista_temp), std(lista_temp)]
            for k in range(len(lista_)):
                lista_[k].append(lista_stat_temp[k])

        for i in range(len(dataset_[0]) - 1):
            massimo = lista_[1][i]
            minimo = lista_[2][i]
            for j in range(len(dataset_)):
                dataset_[j][i] = (dataset_[j][i] - minimo) / (massimo - minimo)

        ##########################################################################
        ##########################################################################

        subject_index = int(file.split(".")[0].split("_")[-1])
        for row in dataset_:
            row.append(subject_index)

        ##########################################################################
        ##########################################################################
        dataset.extend(dataset_)
    return dataset
#######################################################################################
#######################################################################################


if __name__ == "__main__":
    lista_file = os.listdir("dataset_out_clean_balanced_with_updated_v_and_a_features/")
    print(lista_file)

    dataset = unioneEnormalizzazione(lista_file, directory="dataset_out_clean_balanced_with_updated_v_and_a_features")

    for row in dataset:
        for element in row[:-1]:
            if element == 1.0:
                print("[INFO] element equal to 1.0: " + str(element))
            elif element > 1.0:
                print("[INFO] element major than 1.0: "+str(element))

    from_dataset_to_csv(dataset, file_path='dataset_norm/dataset_norm_clean_balanced_with_updated_v_and_a_features_without_test_3.csv')
#######################################################################################
#######################################################################################
