channels = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
bands = ['d', 't', 'a', 'b', 'g']
statistics_features = ['m', 'd']
arousal_and_valence_features = ['v1', 'v2', 'v3', 'v4', 'a1', 'a2', 'a3', 'a4']

labels = []
for channel in channels:
    for band in bands:
        for statistic_feature in statistics_features:
            label = channel + '_' + band + '_' + statistic_feature
            labels.append(label)

labels += arousal_and_valence_features
