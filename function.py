###############################################################
###############################################################
from scipy.signal import freqz
import copy
from scipy.signal import butter, lfilter
from numpy import var, std,  mean
from math import sqrt
from scipy.fft import fft
import function as signal
from utility import mat as m
import csv
from math import log
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch
###############################################################
###############################################################


def integ(x,y,amp_step):
    sum=0
    for i in range(1,int(len(x)/amp_step)):
        if (i*amp_step<len(x)):
            hmean=(y[amp_step*i]+y[amp_step*(i-1)])/2
            sum+=(x[amp_step*i]-x[amp_step*(i-1)])*hmean
    return sum
###############################################################


def statistic(x,y):

    variance=var(y)
    Std=std(y)
    Mean=mean(y)
    sum = 0

    for i in range(len(y)):
        sum+=(y[i]**2)

    sum=sum/len(y)
    root_mean_square=sqrt(sum)
    power=integ(x,y,amp_step=1)

    return power,root_mean_square,Mean,variance,Std
###############################################################


def segment(y,amp,sovr):
    step=amp*128
    new_x=[]
    for i in range(len(y)):
        index=i*int((1-sovr)*step)
        if (index+step<len(y)):
            new_x.append(y[index:index+step])
    return new_x
###############################################################


def butter_highpass_filter(data, order=5):
    b, a = butter(order, 0.16, btype='highpass')
    y = lfilter(b, a, data)
    return y
###############################################################


def iir_filter(data, ns):
    IIR_TC = 256
    back = data[0]
    for i in range(1, ns):
        back = (back * (IIR_TC-1) + data[i]) / IIR_TC
        data[i] = data[i] - back
    return data
###############################################################


def scientific_notation(number):
    if number < 1:
        cnt = 0

        while number < 1:
            number *= 10
            cnt -= 1

        return number, cnt

    elif number > 1:
        cnt = 0

        while number > 1:
            number *= 0.1
            cnt += 1

        return number, cnt

    return number, 1
###############################################################


def butter_bandpass(lowcut, highcut, fs, order=5):
       nyq = 0.5 * fs
       low = lowcut / nyq
       high = highcut / nyq
       b, a = butter(order, [low, high], btype='band')
       return b, a
###############################################################


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
       b, a = butter_bandpass(lowcut, highcut, fs, order=order)
       y = lfilter(b, a, data)
       return y
###############################################################


def box_plot(result, channel_list, bands, show=False):

    plts = []

    for j in range(len(channel_list)):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(22, 8))

        bplot1 = axes[0][0].boxplot([result[0][j+1][2][:],result[0][j+1][4][:]],
                                 vert=True,  # vertical box aligmnent
                                 patch_artist=True,# fill with color
                                 showfliers=False)
        bplot2 = axes[0][1].boxplot([result[1][j+1][2][:],result[1][j+1][4][:]],
                                 vert=True,
                                 patch_artist=True,
                                 showfliers=False)
        bplot3 = axes[0][2].boxplot([result[2][j+1][2][:],result[2][j+1][4][:]],
                                 vert=True,
                                 patch_artist=True,
                                 showfliers=False)
        bplot4 = axes[0][3].boxplot([result[3][j+1][2][:],result[3][j+1][4][:]],
                                 vert=True,
                                 patch_artist=True,
                                 showfliers=False)
        bplot5 = axes[0][4].boxplot([result[4][j+1][2][:], result[4][j+1][4][:]],
                                 vert=True,
                                 patch_artist=True,
                                 showfliers=False)
        bplot6 = axes[1][0].boxplot([result[0][j+1][3][:],result[0][j+1][5][:]],
                                 vert=True,
                                 patch_artist=True,
                                 showfliers=False)
        bplot7 = axes[1][1].boxplot([result[1][j+1][3][:],result[1][j+1][5][:]],
                                 vert=True,
                                 patch_artist=True,
                                 showfliers=False)
        bplot8 = axes[1][2].boxplot([result[2][j+1][3][:],result[2][j+1][5][:]],
                                 vert=True,
                                 patch_artist=True,
                                 showfliers=False)
        bplot9 = axes[1][3].boxplot([result[3][j+1][3][:],result[3][j+1][5][:]],
                                 vert=True,
                                 patch_artist=True,
                                 showfliers=False)
        bplot10 = axes[1][4].boxplot([result[4][j+1][3][:],result[4][j+1][5][:]],
                                 vert=True,
                                 patch_artist=True,
                                 showfliers=False)

        # fill with colors
        colors = ['lightgreen','pink']
        for bplot in (bplot1, bplot2, bplot3, bplot4,bplot5, bplot6, bplot7, bplot8,bplot9,bplot10):
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        # adding horizontal grid lines
        c=0
        for ax in axes[0]:
            ax.yaxis.grid(True)
            ax.set_xticks([y + 1 for y in range(2)], )
            ax.set_xlabel('Class')
            ax.set_ylabel('ESD ('+str(bands[c][0])+')')
            c+=1

        c=0
        for ax in axes[1]:
            ax.yaxis.grid(True)
            ax.set_xticks([y + 1 for y in range(2)], )
            ax.set_xlabel('Class')
            ax.set_ylabel('PSD ('+str(bands[c][0])+')')
            c+=1

        # add x-tick labels
        plt.setp(axes[0], xticks=[y + 1 for y in range(2)],
                 xticklabels=['NoStress','Stress'])

        plt.setp(axes[1], xticks=[y + 1 for y in range(2)],
                 xticklabels=['NoStress','Stress'])

        if show:
            plt.show()

        plt.close()

        plts.append(plt)

    return plts
###############################################################


def from_dataset_to_csv(dataset,file_path):

    f = open(file_path, 'w')

    for i in range(len(dataset)):
        for j in range(len(dataset[i]) - 1):
            f.write(str(dataset[i][j]) + ",")
        f.write(str(dataset[i][-1]) + "\n")
    f.close()

    return None
###############################################################


def from_dicts_to_csv(metrics_dicts, labels, file_path):

    f = open(file_path, 'w')
    metrics_tensor = list()
    metrics_tensor.append(labels)

    for metrics_dict in metrics_dicts:
        row = list()
        for key in metrics_dict:
            row.append(metrics_dict[key])
        metrics_tensor.append(row)

    for i in range(len(metrics_tensor)):
        for j in range(len(metrics_tensor[i]) - 1):
            f.write(str(metrics_tensor[i][j]) + ",")
        f.write(str(metrics_tensor[i][-1]) + "\n")
    f.close()

    return None
###############################################################
def PSD(signal, fs=1.0, plot=False):

    """
    funzionamento analogo a "periodigram".
    "fft" indica la trasformata di Fourier.
    """

    signalfft = fft(signal)
    N = len(signal)
    T = N / fs
    signalfft = signalfft[1:N // 2 +1]
    signalpsd = 2*(1 / (128 * N)) * np.abs(signalfft)**2
    freq = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    if plot:
        plt.plot(freq,signalpsd)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.grid()
        plt.show()

    return freq, signalpsd
###############################################################

def filter_and_process(dataset, channel_list, array=True):
    # Filter and process a noisy signal.
    filter_signals = []

    for i in range(len(channel_list)):
        nsamples = len(dataset)
        #T=nsamples/128
        #t = np.linspace(0, T, int(nsamples), endpoint=False)
        filter_signal = signal.butter_highpass_filter(m.transpmat(m.col(dataset,i)), 2)
        filter_signal = signal.iir_filter(filter_signal, nsamples)
        filter_signal = filter_signal[1:]

        #for k in range(len(filter_signal)):#abbasso a valore assoluto 100 i valori eccessivi
        #    if filter_signal[k] >= 100.0:
        #        filter_signal[k] = 100.0
        #    elif filter_signal[k] <= - 100.0:
        #        filter_signal[k] = - 100.0
        #    else:
        #        pass

        filter_signal=np.reshape(filter_signal, [len(filter_signal), ])
        filter_signals.append(filter_signal)

    if array:
        filter_signals = np.array(filter_signals)
        filter_signals = np.reshape(filter_signals, [len(filter_signals), len(filter_signals[0])])

    return filter_signals
###############################################################


def bands_separation(signals, bands, fs):

    bands_signals = [["Delta"],["Theta"],["Alfa"],["Beta"],["Gamma"]]

    for sign in signals:

        for band in bands:
            lowcut = band[1][0]
            highcut = band[1][1]
            band_signal = signal.butter_bandpass_filter(sign, lowcut, highcut, fs, order=band[2])
            for element in bands_signals:
                if element[0]==band[0]:
                    element.append(band_signal)

    return bands_signals
###############################################################


def raw_data_from_csv(file_path, temporal_range, channel_list=None):

    start, stop = temporal_range

    with open(file_path, newline="", encoding="ISO-8859-1") as filecsv:
        lettore = csv.reader(filecsv, delimiter=",")
        raw_data = []

        for row in lettore:
            raw_data.append(row)

    raw_data = raw_data[1:]
    raw_data = raw_data[int(start * 128):int(stop * 128)]

    signals = []

    for row in raw_data:
        row = row[2:16]

        for i in range(0, len(row)):
            row[i] = float(row[i])

        signals.append(row)

    for row in signals:

        for i in range(len(row)):
            row[i] = float(row[i])

    cut_signals = copy.deepcopy(signals)

    if channel_list != None:

        cut_signals = []

        for j in range(len(signals)):
            row = []

            for i in channel_list:
                row.append(signals[j][i])

            cut_signals.append(row)

    # transform in array and transpose
    signals = copy.deepcopy(cut_signals)
    transp_signals = copy.deepcopy(cut_signals)

    signals = np.reshape(signals, [len(signals), len(channel_list)])
    transp_signals = np.reshape(transp_signals, [len(transp_signals), len(channel_list)])

    transp_signals = np.transpose(transp_signals)

    return signals, transp_signals
###############################################################

def from_energy_to_power(bands_energy):

    bands_power = [["Delta"],["Theta"],["Alfa"],["Beta"],["Gamma"]]

    bands = [["Delta", [0,4]],
           ["Theta", [4,8]],
           ["Alfa", [8,14]],
           ["Beta", [14,30]],
           ["Gamma", [30,40]]]

    for i in range(len(bands_energy)):

        for j in range(1,len(bands_energy[i])):
            bands_power[i].append(signal.segment(bands_energy[i][j], 1, 0.8))#finestre di 1s con sovrapposizione dell'80%

            for k in range(len(bands_power[i][j])):
                f, psd = periodogram(bands_power[i][j][k], 128)
                indexlow = bands[i][1][0]#indice relativo a frequenza bassa della banda
                indexhig = bands[i][1][1]#indice relativo a frequenza alta della banda
                f = f[indexlow:indexhig+1]
                psd = psd[indexlow:indexhig+1]

                for l in range(len(psd)):
                    psd[l] = psd[l].real

                bands_power[i][j][k] = signal.statistic(f, psd)

                nseg = len(bands_power[i][j])

    return bands_power, nseg
###############################################################


def from_signals_to_dataset(bands_energy, bands_power, nseg, channel_list, bands, time_intervals, targets, asymmetry=False):

    data=[]
    data_complete=[]#datasetcompleto
    dataset_complete=[]#datasetcompleto

    for k in range(nseg):
        sample = []

        for j in range(1,len(channel_list)+1):
            for i in range(0,len(bands_power)):
                # solo due caratteristiche statistiche
                # 'mean' e 'std'
                #data.append(sample)
                sample.extend([bands_power[i][j][k][2],bands_power[i][j][k][4]])

        data.append(sample)
        data_complete.append(sample)

    result = []
    for band in bands:
        result.append([band[0], ])

    for i in range(len(result)):

        for channel in channel_list:
            result[i].append([])

        for j in range(1, len(result[i])):

            for k in range(6):
                result[i][j].append([])

            for l in range(len(bands_power[i][j])):
                result[i][j][0].append(bands_power[i][j][l][0])
                result[i][j][1].append(bands_power[i][j][l][2])
    #####################################################################################################################
    ####################################AGGIUNGO CARATTERISTICHE ASIMMETRIA##############################################

    # index_channel*10+index_banda*2+(0-->media,1-->deviazione standard)

    if asymmetry:

        for i in range(len(data_complete)):
            sample = data[i]
            alfa_AF7 = (sample[3 * 10 + 2 * 2 + 0] + sample[2 * 10 + 2 * 2 + 0]) / 2
            alfa_AF8 = (sample[10 * 10 + 2 * 2 + 0] + sample[11 * 10 + 2 * 2 + 0]) / 2
            beta_AF7 = (sample[3 * 10 + 3 * 2 + 0] + sample[2 * 10 + 3 * 2 + 0]) / 2
            beta_AF8 = (sample[10 * 10 + 3 * 2 + 0] + sample[11 * 10 + 3 * 2 + 0]) / 2

            v1 = alfa_AF8 / beta_AF8 - alfa_AF7 / beta_AF7
            v2 = log(alfa_AF7) - log(alfa_AF8)
            v3 = beta_AF7 / alfa_AF7 - beta_AF8 / alfa_AF8
            v4 = alfa_AF8 - alfa_AF7
            a1 = (alfa_AF7 + alfa_AF8) / (beta_AF7 + beta_AF8)
            a2 = -(log(alfa_AF7) + log(alfa_AF8))
            a3 = log((beta_AF7 + beta_AF8) / (alfa_AF7 + alfa_AF8), 2)
            a4 = (beta_AF7 + beta_AF8) / (alfa_AF7 + alfa_AF8)

            data_complete[i].extend([v1, v2, v3, v4, a1, a2, a3, a4])
    #####################################################################################################################
    ############################################ETICHETTATURA DEI DATI###################################################

    # dividere in classi (classificazione binaria)
    # forma --> tempo0.tempo1,tempo2.tempo3,...,tempo(N-1).tempoN

    contrazione = len(bands_power[0][1]) / len(bands_energy[0][1])  # contrazione dovuta a divisione in segmenti
    #time_intervals = input("Inserire intervalli temporali: ")
    #targets = input("Inserire etichette: ")
    targets = targets.split(",")
    time_intervals = time_intervals.split(".")

    for i in range(len(time_intervals)):
        time_intervals[i] = time_intervals[i].split(",")

    for i in range(len(time_intervals)):
        indexl = int(int(time_intervals[i][0]) * 128 * contrazione)
        indexh = int(int(time_intervals[i][1]) * 128 * contrazione)
        for j in range(len(bands)):
            for k in range(len(channel_list)):
                if (targets[i] == '0'):
                    result[j][k + 1][2].extend(result[j][k + 1][0][indexl:indexh])
                    result[j][k + 1][3].extend(result[j][k + 1][1][indexl:indexh])
                if (targets[i] == '1'):
                    result[j][k + 1][4].extend(result[j][k + 1][0][indexl:indexh])
                    result[j][k + 1][5].extend(result[j][k + 1][1][indexl:indexh])
    #####################################################################################################################
    ##########################AGGIUNGO ETICHETTE PER DATASET ADDESTRAMENTO E VALIDAZIONE DELLA RETE######################
        print(len(data_complete))
        print(indexl, indexh)
        if targets[i] == '0':
            for index in range(indexl, indexh):
                data_complete[index].append(0)  # datasetcompleto
                dataset_complete.append(data_complete[index])  # datasetcompleto
        if targets[i] == '1':
            for index in range(indexl, indexh):
                data_complete[index].append(1)  # datasetcompleto
                dataset_complete.append(data_complete[index])  # datasetcompleto
    #####################################################################################################################

    return dataset_complete, result
###############################################################
"""
def from_signals_to_dataset(bands_energy, bands_power, nseg, channel_list, bands, asymmetry=False):

    data=[]
    data_complete=[]#datasetcompleto
    dataset_complete=[]#datasetcompleto

    for k in range(nseg):
        sample = []

        for j in range(1,len(channel_list)+1):
            for i in range(0,len(bands_power)):
                # solo due caratteristiche statistiche
                # 'mean' e 'std'
                #data.append(sample)
                sample.extend([bands_power[i][j][k][2], bands_power[i][j][k][4]])

        data.append(sample)
        data_complete.append(sample)

    result = []
    for band in bands:
        result.append([band[0], ])

    for i in range(len(result)):

        for channel in channel_list:
            result[i].append([])

        for j in range(1, len(result[i])):

            for k in range(6):
                result[i][j].append([])

            for l in range(len(bands_power[i][j])):
                result[i][j][0].append(bands_power[i][j][l][0])
                result[i][j][1].append(bands_power[i][j][l][2])
    #####################################################################################################################
    ####################################AGGIUNGO CARATTERISTICHE ASIMMETRIA##############################################

    # index_channel*10+index_banda*2+(0-->media,1-->deviazione standard)

    if asymmetry:

        for i in range(len(data_complete)):
            print(i)###########################
            sample = data[i]
            print(sample)###########################
            alfa_AF7 = sample[3 * 10 + 2 * 2 + 0]
            alfa_AF8 = sample[10 * 10 + 2 * 2 + 0]
            beta_AF7 = sample[3 * 10 + 3 * 2 + 0]
            beta_AF8 = sample[10 * 10 + 3 * 2 + 0]

            v1 = alfa_AF8 / beta_AF8 - alfa_AF7 / beta_AF7
            v2 = log(alfa_AF7) - log(alfa_AF8)
            v3 = beta_AF7 / alfa_AF7 - beta_AF8 / alfa_AF8
            v4 = alfa_AF8 - alfa_AF7
            a1 = (alfa_AF7 + alfa_AF8) / (beta_AF7 + beta_AF8)
            a2 = -(log(alfa_AF7) + log(alfa_AF8))
            a3 = log((beta_AF7 + beta_AF8) / (alfa_AF7 + alfa_AF8), 2)
            a4 = (beta_AF7 + beta_AF8) / (alfa_AF7 + alfa_AF8)

            data_complete[i].extend([v1, v2, v3, v4, a1, a2, a3, a4])
    #####################################################################################################################
    ############################################ETICHETTATURA DEI DATI###################################################

    # dividere in classi (classificazione binaria)
    # forma --> tempo0.tempo1,tempo2.tempo3,...,tempo(N-1).tempoN

    contrazione = len(bands_power[0][1]) / len(bands_energy[0][1])  # contrazione dovuta a divisione in segmenti
    intervalli_temporali = input("Inserire intervalli temporali: ")
    target = input("Inserire etichette: ")
    target = target.split(",")
    intervalli_temporali = intervalli_temporali.split(".")

    for i in range(len(intervalli_temporali)):
        intervalli_temporali[i] = intervalli_temporali[i].split(",")

    for i in range(len(intervalli_temporali)):
        indexl = int(int(intervalli_temporali[i][0]) * 128 * contrazione)
        indexh = int(int(intervalli_temporali[i][1]) * 128 * contrazione)
        for j in range(len(bands)):
            for k in range(len(channel_list)):
                if (target[i] == '0'):
                    result[j][k + 1][2].extend(result[j][k + 1][0][indexl:indexh])
                    result[j][k + 1][3].extend(result[j][k + 1][1][indexl:indexh])
                if (target[i] == '1'):
                    result[j][k + 1][4].extend(result[j][k + 1][0][indexl:indexh])
                    result[j][k + 1][5].extend(result[j][k + 1][1][indexl:indexh])
    #####################################################################################################################
    ##########################AGGIUNGO ETICHETTE PER DATASET ADDESTRAMENTO E VALIDAZIONE DELLA RETE######################

        if target[i] == '0':
            for index in range(indexl, indexh):
                data_complete[index].append(0)  # datasetcompleto
                dataset_complete.append(data_complete[index])  # datasetcompleto
        if target[i] == '1':
            for index in range(indexl, indexh):
                data_complete[index].append(1)  # datasetcompleto
                dataset_complete.append(data_complete[index])  # datasetcompleto
    #####################################################################################################################

    return dataset_complete, result
###############################################################
"""

"""
f, psd_per = periodogram(result_band_energy[i][1], 128)

f, psd_wel = welch(signal, 128,nperseg=len(signal))

plt.plot(f, np.abs(psd_wel))
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.grid()
plt.show()
"""
###############################################################


def signal_plot(y_values, x_label, y_label, title, start_time, stop_time, output_path):

    y = y_values[start_time*128:stop_time*128]
    nsamples = len(y)
    T = nsamples / 128
    t = np.linspace(0, T, int(nsamples), endpoint=False)

    plt.figure(figsize=(15,5))
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(output_path)

    plt.close()

    return plt


def signal_plot_confronto(first_y_values, second_y_values, x_label, y_label, title, start_time, stop_time, output_path):

    y_1 = first_y_values[start_time*128:stop_time*128]
    y_2 = second_y_values[start_time*128:stop_time*128]

    nsamples = len(y_1)
    T = nsamples / 128
    t = np.linspace(0, T, int(nsamples), endpoint=False)

    plt.plot(t, y_1)
    plt.plot(t, y_2)
    plt.legend(['noisy signal', 'filtered signal'], loc='upper left')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(output_path)

    plt.close()

    return plt
###############################################################


def psd_plot(y_values, x_label='frequency [Hz]', y_label='PSD [V**2/Hz]', title="Power Spectral Density", fs=128, output_path=""):

    f, psd_per = periodogram(y_values, fs)

    plt.plot(np.abs(f), np.abs(psd_per))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(output_path)
    plt.close()
    return plt
###############################################################


# dimostrazione applicazione filtro passabanda su segnale costruito con funzioni sinusoidali
# verifica dell'efficacia dell'ordine utilizzato per il filtro passabanda
if __name__ == "__main__":

       # Sample rate and desired cutoff frequencies (in Hz).
       fs = 128.0
       lowcut = 35
       highcut = 45.0

       # Plot the frequency response for a few different orders.
       plt.figure(1)
       plt.clf()
       for order in [3, 6, 9]:
           b, a = butter_bandpass(lowcut, highcut, fs, order=order)
           w, h = freqz(b, a, worN=256)
           plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

       plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
                '--', label='sqrt(0.5)')
       plt.xlabel('Frequency (Hz)')
       plt.ylabel('Gain')
       plt.grid(True)
       plt.legend(loc='best')

       # Filter a noisy signal.
##########################################################
       T = 1.0
       nsamples = T * fs
       t = np.linspace(0, T, int(nsamples), endpoint=False)
##########################################################
       #a = 0.02
       #f0 = 600.0
       x = 0.1 * np.sin(2 * np.pi * 2.5 * t)
       x += 0.05 * np.sin(2 * np.pi * 5.0 * t)
       x += 0.03 * np.cos(2 * np.pi * 10.0 * t)
       x += 0.02 * np.cos(2 * np.pi * 20.0 * t)
       x += 0.01 * np.cos(2 * np.pi * 40.0 * t)
##########################################################
       plt.figure(2)
       plt.clf()
       plt.plot(t, x, label='Noisy signal')
       plt.xlabel('time (seconds)')
       #plt.hlines([-a, a], 0, T, linestyles='--')
       plt.grid(True)
       plt.axis('tight')
       plt.legend(loc='upper left')

       plt.figure(3)
       plt.clf()
       y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6 )
       plt.plot(t, y, label='Filtered signal')
       plt.xlabel('time (seconds)')
       # plt.hlines([-a, a], 0, T, linestyles='--')
       plt.grid(True)
       plt.axis('tight')
       plt.legend(loc='upper left')
       print(x)
       plt.show()
###############################################################
###############################################################