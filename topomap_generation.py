import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# definisci i nomi dei canali
ch_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
# definisci i tempi di inizio e fine delle fasi
rest_start, rest_stop = 50, 250
test_start, test_stop = 800, 1300
# definisci le bande di frequenza per il filtraggio
freq_bands = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 45]}
sfreq = 128
epochs = []
######################################################################################################################################################################
# crea un ciclo per caricare i dati CSV e creare un oggetto Epochs per ciascun file
for i in range(0, 1):
    # carica il file CSV
    filepath = f'dataset_in/test_{i}.csv'
    data = pd.read_csv(filepath, header=1)
    # seleziona le colonne specificate
    data = data.iloc[:, 2:16]
    # rinomina le colonne con i nomi dei canali
    data.columns = ch_names
    #print(data)
######################################################################################################################################################################
    # crea i file RAW per la funzione make_fixed_length_events
    raw = mne.io.RawArray(data.T.values, info=mne.create_info(ch_names, sfreq, ch_types='eeg'))
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    # crea gli eventi per la fase di riposo e la fase di test
    rest_events = mne.make_fixed_length_events(raw, id=1, start=rest_start,
                                                stop=rest_stop, duration=(rest_stop - rest_start),
                                                overlap=0, first_samp=0)
    test_events = mne.make_fixed_length_events(raw, id=2, start=test_start,
                                                stop=test_stop, duration=(test_stop - test_start),
                                                overlap=0, first_samp=0)
    events = np.concatenate([rest_events, test_events], axis=0) #restituisce un array 1D contenente 3 valori: il primo rappresenta 
                                                                #il punto di start espresso in termini dell'indice del campione
                                                                #ad esempio se lo start per il rest è a 50s allora 50x128=6400 sarà
                                                                #lo starting point espresso in termini di indice del campione
                                                                #il secondo termine rappresenta 
                                                                #il terzo termine rappresenta l'ID dell'evento
    #print(events)
######################################################################################################################################################################
    # crea l'oggetto Epochs utilizzando le colonne selezionate e gli eventi creati
    epoch1 = mne.Epochs(raw, events, event_id={'rest': 1},
                        tmin=rest_start, tmax=rest_stop, baseline=None, preload=True)
    epoch2 = mne.Epochs(raw, events, event_id={'test': 2},
                        tmin=test_start, tmax=test_stop, baseline=None, preload=True)
    epochs.append(epoch1)
    epochs.append(epoch2)
    #info_data1 = epoch1.get_data()
    #info_data2 = epoch2.get_data()
    #print(info_data1.shape)
    #print(info_data2.shape)
    #epoch1.plot_psd()
    #epoch2.plot_psd()
    #plt.show()
######################################################################################################################################################################
    #bisogna unire le epoche?
    #concatenated_epochs = mne.concatenate_epochs(all_epochs, add_offset=True)
######################################################################################################################################################################
    #filtra i dati per ogni banda di frequenza e calcola le mappe topografiche
    for j, epoch in enumerate(epochs):
        if j == 0:
            for band_name, band_freqs in freq_bands.items():
                # filtra i dati per la banda di frequenza corrente
                epoch_band = epoch.copy().filter(band_freqs[0], band_freqs[1])
                #info_band=epoch1_band.get_data()
                #print(info_band.shape)
                fig=epoch_band.plot_psd(show=False)
                #fig.suptitle(f'subj {i}, negative stimulus - {band_name} band', y=11)
                fig.savefig(f'topomap/subj_{i}_{band_name}_psd_rest.png')
        else:
            for band_name, band_freqs in freq_bands.items():
                # filtra i dati per la banda di frequenza corrente
                epoch_band = epoch.copy().filter(band_freqs[0], band_freqs[1])
                #info_band=epoch1_band.get_data()
                #print(info_band.shape)
                fig=epoch_band.plot_psd(show=False)
                #fig.suptitle(f'subj {i}, postive stimulus - {band_name} band')
                fig.savefig(f'topomap/subj_{i}_{band_name}_psd_test.png')
            '''
            # calcola le mappe topografiche per la banda di frequenza corrente
            evoked_band = epoch2_band.average()
            evoked_info = evoked_band.get_data()
            print(evoked_info.shape)
            #fig = evoked_band.plot_topomap(show=False, ch_type="eeg", res=128, average=50, colorbar=False, cbar_fmt="%.1f", time_format="%.1f")
            #fig.suptitle(f'subj {i} - {band_name} band')
            #fig.savefig(f'topomap/subj_{i}_{band_name}_rest.png')
            '''