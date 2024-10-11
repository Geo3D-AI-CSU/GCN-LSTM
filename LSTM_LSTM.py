import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from method import createXY,rmse

# Define the LSTM_LSTM model architecture
def build_model(optimizer,neurons,learn_rate):
    """
    Build an LSTM_LSTM model for time series prediction.

    Parameters:
    - optimizer (str): The optimization algorithm for training the model (e.g., 'adam').
    - neurons (int): The number of LSTM units in each layer (affects model complexity).
    - learn_rate (float): The learning rate for the optimizer.
    - epochs (int): The number of training epochs.

    Returns:
    - grid_model (Sequential): The compiled LSTM_LSTM model.
    """
    grid_model = Sequential()
    grid_model.add(LSTM(neurons,return_sequences=True,input_shape=(3,1)))
    grid_model.add(LSTM(neurons))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model


if __name__ == '__main__':
    outIndex = r'.\result\LSTM_LSTM_Index.csv'
    outPrediction = r'.\result\LSTM_LSTM_Prediction.csv'
    outTotalIndex = r'.\result\LSTM_LSTM_Total_Index.csv'

    outDataArr = []
    totalPrediction = []
    totalOriginal = []
    outPredictionArr = []
    outSingleOriginalArr = []
    outSinglePredictionArr = []
    outTotalIndexArr = []
    for i in range(0, 16):
        data = pd.read_excel(r'./data/well-all-Original.xls', sheet_name='Sheet1', index_col=[0])
        df = data.iloc[:, [i]]

        # Split the data into training and testing sets
        df_for_training = df[:-216]
        df_for_testing = df[-216:]
        print(df_for_training.shape)
        print(df_for_testing.shape)
        # Data normalization
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_for_training_scaled = scaler.fit_transform(df_for_training)
        df_for_testing_scaled = scaler.transform(df_for_testing)
        # Create input sequences (X) and output labels (Y) for the model
        trainX, trainY = createXY(df_for_training_scaled, 3)
        testX, testY = createXY(df_for_testing_scaled, 3)

        # Create and train the model
        grid_model = KerasRegressor(build_fn=build_model, verbose=1)
        parameters = {'batch_size': [64],
                      'epochs': [150],
                      'optimizer': ['adam'],
                      'neurons': [64],
                      'learn_rate': [0.001]
                      }
        grid_search = GridSearchCV(estimator=grid_model,
                                   param_grid=parameters,
                                   cv=2)
        grid_search = grid_search.fit(trainX, trainY, validation_data=(testX, testY))
        grid_search.best_params_
        my_model = grid_search.best_estimator_.model
        # #Save the model
        my_model.save(r"./model/LSTM_LSTM.h5")

        prediction = my_model.predict(testX)

        # Repeat the predictions to match the original data format
        prediction_copies_array = np.repeat(prediction, 2, axis=-1)

        # Inverse transform the predictions and original data to their original scale
        pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(prediction), 2)))[:, 0]
        original_copies_array = np.repeat(testY, 2, axis=-1)
        original = scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), 2)))[:, 0]

        # total Prediction
        pred_list = pred.tolist()
        original_list = original.tolist()
        j = 0
        for j in range(len(pred_list)):
            outPredictionData = pred_list[j]
            outOriginalData = original_list[j]
            totalPrediction.append(outPredictionData)
            totalOriginal.append(outOriginalData)
            Abs = abs(pred_list[j] - original_list[j])
            outSinglePredictionData = [pred_list[j], original_list[j], Abs]
            outPredictionArr.append(outSinglePredictionData)

        print("pred_list--", pred_list)
        print("original_list--", original_list)
        outSinglePredictionArr.append(pred_list)
        outSingleOriginalArr.append(original_list)

        r2 = r2_score(pred, original)
        mse = mean_squared_error(pred, original)
        Rmse = rmse(pred, original)
        mae = mean_absolute_error(pred, original)

        print("Pred Values-- ", pred)
        print("\nOriginal Values-- ", original)
        print("r2", r2_score(pred, original))
        print("mse", mean_squared_error(pred, original))
        print("rmse", rmse(pred, original))
        print("mae", mean_absolute_error(pred, original))

        # Output evaluation metrics
        outdata = ['Well-{}'.format(i + 1), r2, mse, Rmse, mae]
        outDataArr.append(outdata)

        # ShowPlot(original,pred)
        plt.figure(figsize=(10, 8))
        font = {
            'family': 'Times New Roman',
            'weight': 'bold',
            'size': 20
        }
        x = pd.date_range('2015-7-5', periods=len(pred), freq='5d')
        plt.plot(x, original, color='red', label='Real  GWL', linewidth=3.0)
        plt.plot(x, pred, color='blue', label='Predicted  GWL', linewidth=3.0)
        plt.title('Well-{}'.format(i + 1), fontdict=font)
        plt.xlabel('Time', fontdict=font)
        plt.ylabel(' GWL', fontdict=font)
        plt.xticks(rotation=45, weight='bold', size=18)
        plt.yticks(weight='bold', size=18)
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.legend(loc='upper left', prop={'size': 18})
        # plt.show()
        plt.savefig('./PNG/LSTM_LSTM-Well-{}.png'.format(i + 1))

    total_r2 = r2_score(totalPrediction, totalOriginal)
    total_mse = mean_squared_error(totalPrediction, totalOriginal)
    total_Rmse = rmse(totalPrediction, totalOriginal)
    total_mae = mean_absolute_error(totalPrediction, totalOriginal)
    # total_rRMSE = rRMSE(Original, Prediction)
    print("Total Index--")
    print("r2", total_r2)
    print("mse", total_mse)
    print("rmse", total_Rmse)
    print("mae", total_mae)
    # Output evaluation metrics
    outIndexdata = [total_r2, total_mse, total_Rmse, total_mae]
    outTotalIndexArr.append(outIndexdata)

    outDataArr = pd.DataFrame(outDataArr)
    outDataArr.rename(columns={0: 'well', 1: 'R2', 2: 'mse', 3: 'rmse', 4: 'mae'}, inplace=True)
    print(outDataArr)
    outDataArr.to_csv(outIndex)

    # Save prediction to a file
    # Total well
    outPredictionArr = pd.DataFrame(outPredictionArr)
    outPredictionArr.rename(columns={0: 'WellPrediction', 1: 'WellOriginal', 2: 'WellAbs',
                                     }, inplace=True)
    print(outPredictionArr)
    outPredictionArr.to_csv(outPrediction)

    # Save evaluation metrics to a file
    outTotalIndexArr = pd.DataFrame(outTotalIndexArr)
    outTotalIndexArr.rename(columns={0: 'R2', 1: 'mse', 2: 'rmse', 3: 'mae', 4: 'rRMSE'}, inplace=True)
    print(outTotalIndexArr)
    outTotalIndexArr.to_csv(outTotalIndex)

