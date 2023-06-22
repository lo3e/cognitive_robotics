######################################################################################
######################################################################################
from function import *
import os
######################################################################################
######################################################################################

fs = 128

channel_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

bands = [["Delta", (0.1, 3.9), 3],
         ["Theta", (4.0, 7.9), 6],
         ["Alfa", (8.0, 13.9), 8],
         ["Beta", (14.0, 29.9), 17],
         ["Gamma", (30.0, 40.0), 18]]

files_paths_list = os.listdir("dataset_in")
times_list = []

file = open("crono_auto_clean.txt", "r")
lines = file.readlines()
file.close()

for line in lines:
    line = line.split("|")[1]
    times_list.append(line)
    print(line)
    
#for file_path, times in zip(files_paths_list, times_list):
for i in range(len(times_list)):

    times = times_list[i]
    for path in files_paths_list:
        if str(i) in path:
            file_path = path
            break

    start = int(times.split(",")[0])
    stop = int(times.split(",")[-1])
    file_path = f"dataset_in/{file_path}"

    print(start, stop)
######################################################################################
# file_path --> path del file csv contenente i dati grezzi
# temporal_range --> tupla di coordinate temporali che definiscono, in secondi, l'intervallo temporale da estrarre '(start, stop)'
# channel_list --> lista dei canali da considerare, espressi in variabili 'int'

    signals, trasp_signals = raw_data_from_csv(file_path=file_path,
                                               temporal_range=(start, stop),
                                               channel_list=channel_list)

    ######################################################################################
    for j in range(len(trasp_signals)):
        signal_plot(y_values=trasp_signals[j], x_label="Time(s)", y_label="voltage(MicroVolt)", title="Raw signal",
                    start_time=200, stop_time=400,
                    output_path=f"signal_plots/raw_signal_plots/raw_signal_sub{i}_channel{j}")
    ######################################################################################

# 'signals' è la matrice rappresentante unicamente i segnali misurati dal caschetto, dunque avendo eliminato etichette e dati superflui
# 'trasp_signals' è semplicemente la trasposta di 'signals'
######################################################################################

######################################################################################
# passa da segnale centrato intorno ad un valore di crica 4200 microvolt ad un segnale centrato in zero (IRR filter)
# l'ampiezza tipica dels egnale è di 200 microvolt
# applica inoltre un filtro passa basso per eliminare eventuali rumori di fondo a bassa frequenza (non imputabili ad attività bioelettriche)

    filter_signals = filter_and_process(signals, channel_list)

    ######################################################################################
    for j in range(len(trasp_signals)):
        signal_plot(y_values=filter_signals[j], x_label="Time(s)", y_label="voltage(MicroVolt)", title="Filtered signal",
                    start_time=200, stop_time=400,
                    output_path=f"signal_plots/filtered_signal_plots/filtered_signal_sub{i}_channel{j}")
    ######################################################################################

# return: 'filter_signals'
# sengnale filtrato e processato in formato di "numpy array"
######################################################################################

######################################################################################
# Sample rate and desired cutoff frequencies (in Hz).
# bands --> lista delle bande con intervalli modificabili e ordine di approssimazione del filtro
# fs --> frequenza di campionamento del dispositivo, fs(Emotiv epoc plus) = 128

    bands_energy = bands_separation(filter_signals, bands, fs)

    ######################################################################################
    for j in range(len(bands_energy)):
        for k in range(1, len(bands_energy[j])):

            signal_plot(y_values=bands_energy[j][k], x_label="Time(s)", y_label="voltage(MicroVolt)", title=f"Signal Band {j}",
                        start_time=200, stop_time=202,
                        output_path=f"signal_plots/bands_signal_plots/bands_signal_sub{i}_channel{k}_band{j}")

            psd_plot(y_values=bands_energy[j][k], output_path=f"signal_plots/PSD_plots/PSD_sub{i}_channel{k}_band{j}")
    ######################################################################################

# return: bands_energy
# segnale diviso in bande
######################################################################################

######################################################################################
# separo in 'nseg' segmenti (finestre temporali) di 1s e sovrapposizione dell'80%
# su ogni finestra temporale (segmento) calcolo la PSD
# per ogni finestra temporale estraggo dalla PSD (che è una distribuzione) le caratteristiche statistiche rilevanti

    bands_power, nseg = from_energy_to_power(bands_energy)

# return: bands_power == power,root_mean_square,Mean,variance,Std !!!!!!!!!!!!(power??)
# return: nseg
# nseg --> numero di segmenti ricavati dalla segmentazione del segnale in finestre temporali (1s con sovrapposizione dell' 80%)
######################################################################################

######################################################################################
# prende caratteristiche statistiche in PSD e le etichetta nel dataset, da fornire poi alla rete per il training
# dataset --> contiene solo le carqatteristiche statistiche necessarie alla rete neurale (media e deviazione standard)
# result --> struttura le caratteristiche statistiche per la generazione del box plot
# asimmetry --> asimmetry == 'True' ==> aggiunge in coda al vettore 'dato', contenuto nella matrice 'dataset', 8 caratteristiche di asimmetria

    dataset, result = from_signals_to_dataset(bands_energy, bands_power, nseg, channel_list, bands,
                                              time_intervals=times,
                                              targets="0,1",
                                              asymmetry=True)

# return: dataset, result
# result --> contenitore di informazioni statistiche da fornire alla funzione 'box_plot' per la rappresentazione grafica
# dataset --> contiene vettori 'dato' di 140 caratteristiche (se asimmetry == False) o 148 caratteristiche (se asimmetry == True)
# oltre alle caratteristiche già citate ogni vettore 'dato' contenuto nel dataset possiede come ultimo elemento il label di classe
# Le caratteristiche nel vettore 'dato' sono indicizzabili nel seguente modo:
#                   index = channel * 10 + banda * 2 + (0/1)
# channel è l'indice relativo al canale (per epoc plus varia tra 0 e 13)
# banda è l'indice relativo alla banda considerata (delta ==0, theta ==1, alpha ==2, beta == 3, gamma == 4)
# l'ultimo termine vale 0 se si desidere selezionare la media, oppure 1 se si vuole ottenere la deviazione standard
######################################################################################

######################################################################################
# genera i boxplot per il confronto visivo dei dati statistici per segnali relativi a diverse classi

    plts = box_plot(result, channel_list, bands)

# return: plts
# plts --> lista di oggetti 'boxplot', uno per ogni canale
######################################################################################

    output_file_path = f"dataset_out/{file_path.replace('test', 'dataset').split('/')[-1]}".replace('.', '_clean.csv')

    from_dataset_to_csv(dataset, file_path=output_file_path)

######################################################################################








