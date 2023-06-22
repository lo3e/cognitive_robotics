from tensorflow import keras
import keras_tuner as kt
import predictive_models
from predictive_models import *
import os
########################################################################################################################
########################################################################################################################


def model_builder(hp):

  global INPUT_SHAPE

  model = keras.Sequential()

  hp_n_layers = hp.Int('n_layers', min_value=1, max_value=3, step=1)
  hp_output_activation = hp.Choice('output_activation', values=['softmax', 'sigmoid'])
  hp_optimizer = hp.Choice('optimizer', values=['adam', 'adamax'])

  for i in range(hp_n_layers):
      hp_units = hp.Int(f'units_{i}', min_value=32, max_value=1024, step=8)
      hp_dropout = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.9, step=0.1)

      if i == 0:
          model.add(keras.layers.Dense(units=hp_units, activation='relu', input_shape=INPUT_SHAPE))
      else:
          model.add(keras.layers.Dense(units=hp_units, activation='relu'))

      model.add(keras.layers.Dropout(hp_dropout))

  if hp_output_activation == 'softmax':
      model.add(keras.layers.Dense(2, activation=hp_output_activation))
      loss = 'sparse_categorical_crossentropy'
  elif hp_output_activation == 'sigmoid':
      model.add(keras.layers.Dense(1, activation=hp_output_activation))
      loss = 'binary_crossentropy'

  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  if hp_optimizer == 'adamax':
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=hp_learning_rate),#hp_learning_rate),
                  loss=loss,
                  metrics=['accuracy'])

  elif hp_optimizer == 'adam':
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),  # hp_learning_rate),
                    loss=loss,
                    metrics=['accuracy'])

  return model
########################################################################################################################


def keras_model_optimization(Xtrain, ytrain, Xval, yval, Xtest, ytest, input_shape, model_builders, model_names):

    global INPUT_SHAPE
    INPUT_SHAPE = input_shape

    if not Xtest:
        Xtest = Xval
        ytest = yval

    result = None

    for batch_size, epochs in [(32, 100)]:

        for model_builder, model_name in zip(model_builders, model_names):

            tuner = kt.Hyperband(model_builder,
                                 objective='val_accuracy',
                                 max_epochs=100,
                                 factor=3,
                                 hyperband_iterations=1,
                                 directory='hyper_parameters_tuning_dir',
                                 project_name=f'model-name-{model_name}_batch-size-{batch_size}_input-shape_{input_shape}_tuning')

            stop_early_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            stop_early_accuracy = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

            start_optimization_time = time.time()

            tuner.search(Xtrain, ytrain, validation_data=(Xval, yval), batch_size=batch_size, epochs=epochs,
                         callbacks=[stop_early_loss, stop_early_accuracy])

            # Get the optimal hyperparameters
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            print(f"""The hyperparameter search is complete. The optimal values are the following:
                  \n    n_layers = {best_hps.get('n_layers')}\n""")

            for i in range(best_hps.get('n_layers')):
                print(f"""    units_{i} = {best_hps.get(f'units_{i}')}\n""")

            for i in range(best_hps.get('n_layers')):
                print(f"""    dropout_{i} = {best_hps.get(f'dropout_{i}')}\n""")

            print(f"""    output_activation = {best_hps.get('output_activation')}\n""")

            print(f"""    optimizer = {best_hps.get('optimizer')}\n""")

            print(f"""    learning_rate = {best_hps.get('learning_rate')}\n""")

            # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
            model = tuner.hypermodel.build(best_hps)
            history = model.fit(Xtrain, ytrain, validation_data=(Xval, yval), batch_size=batch_size, epochs=epochs,
                                callbacks=[stop_early_loss, stop_early_accuracy])

            eval_result = model.evaluate(Xval, yval)

            print("[test loss, test accuracy]:", eval_result)

            val_acc_per_epoch = history.history['val_accuracy']
            best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
            print(f'Best epoch: {best_epoch}')

            hypermodel = tuner.hypermodel.build(best_hps)

            # Retrain the model
            hypermodel.fit(Xtrain, ytrain, validation_data=(Xval, yval), batch_size=batch_size, epochs=best_epoch)

            eval_result = hypermodel.evaluate(Xval, yval)
            print("[test loss, test accuracy]:", eval_result)

            # diamo in output i risultati

            predictions = hypermodel.predict(Xtest, batch_size=8)

            if best_hps.get('output_activation') == "sigmoid":
                predictions = map(lambda x: x.round(0), predictions)
            elif best_hps.get('output_activation') == "softmax":
                predictions = map(lambda x: x.argmax(), predictions)

            predictions = list(predictions)

            report = classification_report(ytest, predictions, target_names=np.array(["NoEngagement", "Engagement"]),
                                           digits=4, output_dict=True)

            print()
            print(report)

            stop_optimization_time = time.time()

            optimization_time = stop_optimization_time - start_optimization_time
            
            # plot result
            try:
                os.mkdir(f"hyper_parameters_tuning_dir/plots_NN/batch_size_{batch_size}")
            except FileExistsError:
                pass

            predictive_models.NN_plot(history, image_file_path=f"hyper_parameters_tuning_dir/plots_NN/batch_size_{batch_size}/{model_name}.png")

            # save model
            try:
                os.mkdir(f"hyper_parameters_tuning_dir/models/batch_size_{batch_size}")
            except FileExistsError:
                pass

            model.save(f"hyper_parameters_tuning_dir/models/batch_size_{batch_size}/{model_name}.h5")

            best_hyperparameters = dict()
            best_n_layers = best_hps.get('n_layers')
            best_hyperparameters['n_layers'] = best_n_layers
            best_hyperparameters['activation_functions'] = list()
            best_hyperparameters['dense_values'] = list()
            best_hyperparameters['drop_values'] = list()
            best_hyperparameters['learning_rate'] = best_hps.get('learning_rate')
            best_hyperparameters['optimizer'] = best_hps.get('optimizer')

            for i in range(best_n_layers):
                best_hyperparameters['dense_values'].append(best_hps.get(f'units_{i}'))
                best_hyperparameters[f'drop_values'].append(best_hps.get(f'dropout_{i}'))
                best_hyperparameters[f'activation_functions'].append('relu')

            best_hyperparameters[f'activation_functions'].append(best_hps.get('output_activation'))

            if result and eval_result[1] > result['loss_and_Accuracy'][1]:
                result = {'model_path': f"hyper_parameters_tuning_dir/models/batch_size_{batch_size}/{model_name}.h5",
                          'model_name': model_name, 'batch_size': batch_size, 'best_model': tuner.hypermodel.build(best_hps),
                          'best_params': best_hyperparameters, 'best_epoch_on_validation': best_epoch,
                          'optimization_time': optimization_time}
            elif not result:
                result = {'model_path': f"hyper_parameters_tuning_dir/models/batch_size_{batch_size}/{model_name}.h5",
                          'model_name': model_name, 'batch_size': batch_size, 'best_model': tuner.hypermodel.build(best_hps),
                          'best_params': best_hyperparameters, 'best_epoch_on_validation': best_epoch,
                          'optimization_time': optimization_time}

    return result
########################################################################################################################
########################################################################################################################


if __name__ == "__main__":

    with open("dataset_norm/dataset_norm.csv", newline="", encoding="ISO-8859-1") as filecsv:
        lettore = csv.reader(filecsv, delimiter=",")
        dataset = []
        for row in lettore:
            dataset.append(row)
    for row in dataset:
        for i in range(len(row)):
            row[i] = float(row[i])

    split_dataset = split_by_subjects(dataset, test_subjects=[], validation_size=0.3)

    Xtrain = split_dataset["Xtrain"]
    Xvalidation = split_dataset["Xvalidation"]
    Xtest = split_dataset["Xtest"]

    ytrain = []
    for elemento in Xtrain:
        ytrain.append(elemento[-1])
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

    print(f"[INFO] data_shape: train = {Xtrain.shape}, validation = {Xvalidation.shape}, test = {Xtest.shape}")
    print(f"[INFO] label_shape: train = {ytrain.shape}, validation = {yvalidation.shape}, test = {ytest.shape}")
    print()

    model_builders = [model_builder]
    model_names = ['MLP']

    result = keras_model_optimization(Xtrain, ytrain, Xvalidation, yvalidation, Xtest, ytest, (148, ), model_builders, model_names)

    print(result)