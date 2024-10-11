from pandas import DataFrame
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
import tensorflow.compat.v1 as tf
from keras.wrappers.scikit_learn import KerasRegressor
import xlrd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from keras.models import load_model
from keras_bert import get_custom_objects
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from spektral_gcn import GraphConv
from spektral_utilities import *
from method import rmse

from scipy.ndimage import gaussian_filter1d
from scipy import stats

# import pywt


def excel_to_matrix(path):
    """
    Load data from an Excel file and convert it into a matrix.

    Parameters:
    - path (str): The file path to the Excel file to be read.

    Returns:
    - datamatrix (numpy.ndarray): The data from the Excel file as a 2D NumPy array (matrix).
    """
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]  # Get the first sheet in the Excel file
    nrows = table.nrows  # Number of rows
    ncols = table.ncols  # Number of columns
    datamatrix = np.zeros((nrows, ncols))
    for x in range(ncols):
        cols = table.col_values(x)
        cols1 = np.matrix(cols)  # Convert the list to a matrix for matrix operations
        datamatrix[:, x] = cols1
    return datamatrix


def createXY2(dataset, n_past):
    """
    Create input sequences and output labels for the model based on a time series dataset.

    Parameters:
    - dataset (numpy.ndarray): The time series dataset for which sequences and labels are created.
    - n_past (int): The number of past time steps (samples) to consider for each sequence.

    Returns:
    - dataX (numpy.ndarray): An array of input sequences (X) with shape (num_samples, n_past, num_features).
    - dataY (numpy.ndarray): An array of output labels (Y) with shape (num_samples, num_features).
    """
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0:dataset.shape[1]])
    return np.array(dataX), np.array(dataY)


def load_gwl_data(dataset):
    """
    Load groundwater level (GWL) data from a specified dataset.

    Parameters:
    - dataset (str): The file path or dataset identifier to load the GWL data from.

    Returns:
    - df (pandas.DataFrame): A pandas DataFrame containing the GWL data.
    """
    gwl_df = pd.read_excel(
        r'./data/well-all-Original.xls',
        sheet_name='Sheet1', index_col=[0])

    df = gwl_df.iloc[:, 0:16]
    return df


# Define the GCN_LSTM model architecture
def get_model():
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Define input of the model
    # Two input layers have been defined, one is inp_ Lap represents an input matrix of (16, 16);
    inp_lap = tf.keras.Input((16, 16))
    # The other one is inp_ Feature, representing an input of (16, 72), is a matrix containing 3 features.
    inp_feat = tf.keras.Input((16, 3))

    # Two GraphConv layers were applied, each with 32 output units, and the tanh activation function was used.
    # These layers are used for graph convolution operations on the feature matrix.
    x = GraphConv(32, activation='tanh')([inp_feat, inp_lap])
    x2 = GraphConv(32, activation='tanh')([inp_feat, inp_lap])
    # Here, flatten the output of two graph convolutional layers and transform them from 2D tensors to 1D vectors.
    x = tf.keras.layers.Flatten()(x)
    x2 = tf.keras.layers.Flatten()(x2)

    # Applied an LSTM layer with 64 units and tanh activation function, returning the entire sequence.
    # Then flatten the output of the LSTM layer
    xx = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True)(inp_feat)
    xx = tf.keras.layers.Flatten()(xx)

    # Connect the outputs of the LSTM layer and three Flatten layers together,
    # and then apply a 10% Dropout layer to prevent overfitting.
    x = tf.keras.layers.Concatenate()([xx, x, x2])
    x = tf.keras.layers.Dropout(0.1)(x)
    print(x.shape)

    # A fully connected layer with 16 units has been defined to generate the output of the network.
    # The number of units in the output layer can be modified according to task requirements.
    out = tf.keras.layers.Dense(16)(x)
    print(out.shape)

    # Finally, the input and output are combined into a model,
    # and MSE (mean squared error) is used as the loss function for training with the Adam optimizer,
    # while MSE is used as the evaluation metric.
    model = tf.keras.Model([inp_lap, inp_feat], out)
    model.compile(optimizer=opt, loss='mse',
                  # metrics='accuracy')
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def rRMSE(y_obs, y_sim):
    sum1 = 0
    for i in range(len(y_obs)):
        sum1 = sum1 + (y_sim.iloc[i] - y_obs.iloc[i]) ** 2
    rRMSE = (sum1 / len(y_obs)) ** 0.5
    return rRMSE


# # Wavelet transform function
# def wavelet_transform(data, wavelet='db1', level=1):
#     coeffs = pywt.wavedec(data, wavelet, level=level)
#     return np.concatenate(coeffs)


# Outlier detection and removal function
def remove_outliers(data, threshold=3.0):
    z_scores = np.abs(stats.zscore(data))
    mask = (z_scores < threshold).all(axis=1)
    return data[mask]


# Gaussian filtering function
def gaussian_filter(data, sigma=1.0):
    return gaussian_filter1d(data, sigma=sigma, mode='reflect')


if __name__ == '__main__':
    # Define the output index file:R2,MSE,RMSE,MAE
    outIndex = r'.\result\GCN-LSTM_Index.csv'
    outPrediction = r'.\result\GCN-LSTM_Total_Well.csv'
    outTotalIndex = r'.\result\GCN-LSTM_Total_Index.csv'
    outSinglePrediction = r'.\result\GCN-LSTM_Total_Well_Single_Prediction.csv'
    outDataArr = []
    outTotalIndexArr = []
    outSinglePredictionArr = []
    outSingleOriginalArr = []
    outPredictionArr = []
    outOriginalArr = []
    totalPrediction = []
    totalOriginal = []

    # Load GWL data
    df = load_gwl_data('gwl')

    # Set the decomposition level of wavelet transform,
    # threshold for outlier handling,
    # and standard deviation parameter (sigma) for Gaussian filtering
    wavelet_level = 3
    outlier_threshold = 4
    sigma_value = 0.5

    # Perform wavelet transform, Gaussian filtering, and outlier detection and removal on each column,
    # And output a comparison chart before and after data processing
    #
    # # Create drawing
    # # fig, axs = plt.subplots(nrows=num_columns, ncols=2, figsize=(12, 3 * num_columns))
    # fig, axs = plt.subplots(nrows=16, ncols=2, figsize=(12, 3 * 16))
    #
    # for i, col in enumerate(df.columns):
    #     # Draw preprocessed image
    #     axs[i, 0].plot(df[col], label='预处理前')
    #     axs[i, 0].set_title(f'{col} - 预处理前')
    #
    #     # # Wavelet transform function
    #     # df[col] = wavelet_transform(df[col], level=wavelet_level)
    #     # Gaussian filter
    #     df[col] = gaussian_filter(df[col], sigma=sigma_value)
    #     # # Abnormal value detection and removal
    #     # df[[col]] = remove_outliers(df[[col]], threshold=outlier_threshold)
    #
    #     # Draw preprocessed image
    #     axs[i, 1].plot(df[col], label='预处理后', color='orange')
    #     axs[i, 1].set_title(f'{col} - 预处理后')
    #
    # # Adjusting Layout
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(
    #         '..\data\PNG\GCN-LSTM-All_Well_gaussian.png')
    #
    # print("预处理后的数据：")
    # print(df.head())

    # Load adjacency matrices (Wij, Cij)
    Wij = excel_to_matrix(
        r'./data/Spatial_adj_tin.xls')
    Cij = excel_to_matrix(
        r'./data/Attribute_Cij.xls')
    Dij = Cij * Wij  # Multiply similarity matrices
    # Divide the data into training and testing sets
    df_for_training = df[:-216]
    df_for_testing = df[-216:]

    for i, col in enumerate(df_for_training.columns):
        # Gaussian filter
        df_for_training[col] = gaussian_filter(df_for_training[col], sigma=sigma_value)

    # Data normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_for_training_scaled = scaler.fit_transform(df_for_training)
    df_for_testing_scaled = scaler.transform(df_for_testing)

    # Create input sequences (X) and output labels (Y) for the model
    # 3 is the step size of the sliding time window
    trainX, trainY = createXY2(df_for_training_scaled, 3)
    testX, testY = createXY2(df_for_testing_scaled, 3)
    # Transpose the data to match the model's input dimensions
    trainX = tf.transpose(trainX, [0, 2, 1])
    testX = tf.transpose(testX, [0, 2, 1])

    # Create the local pooling filter based on Dij;
    # Dij is the product of a spatial self similarity matrix Wij and an attribute self similarity matrix Cij
    X_train_lap = localpooling_filter(Dij)
    # Obtain the number of samples in the training set
    num_samples = trainX.shape[0]
    # Use the np. file function to replicate the local pooling filter on the sample dimension,
    # ensuring that it matches the shape of the training set.
    X_train_lap = np.tile(X_train_lap, (num_samples, 1, 1))
    X_test_lap = localpooling_filter(Dij)
    num_samples = testX.shape[0]
    X_test_lap = np.tile(X_test_lap, (num_samples, 1, 1))

    # # Create and train the model
    grid_model = KerasRegressor(build_fn=get_model, verbose=1)
    grid_model.fit([X_train_lap, trainX], trainY, epochs=150, batch_size=64,
                   validation_data=([X_test_lap, testX], testY))

    # # Save the model
    grid_model.model.save(
        r"./model/GCN_LSTM.h5")

    # Make predictions
    prediction = grid_model.predict([X_test_lap, testX])

    # Inverse transform the predictions and original data
    pred = scaler.inverse_transform(prediction)
    original = scaler.inverse_transform(testY)

    k = 0
    # Evaluate the model and display result of each well
    for i in range(0, 16):
        print("\nPred type-- ", type(pred))
        print("\nPred -- ", pred)
        pred_well = pred[:, i]
        # pred_well type--  <class 'numpy.ndarray'>
        print("\npred_well type-- ", type(pred_well))
        original_well = original[:, i]
        print("\nPred Values-- ", i)
        print(pred_well)
        print("\nOriginal Values-- ", i)
        print(original_well)

        predDf = pd.DataFrame(pred_well)
        originalDf = pd.DataFrame(original_well)
        r2 = r2_score(original_well, pred_well)
        mse = mean_squared_error(pred_well, original_well)
        Rmse = rmse(pred_well, original_well)
        mae = mean_absolute_error(pred_well, original_well)
        rrmse = rRMSE(originalDf, predDf)
        print("r2", r2)
        print("mse", mse)
        print("rmse", Rmse)
        print("mae", mae)
        print("rrmse", rrmse)

        # ShowPlot(original,pred)
        plt.figure(figsize=(10, 8))
        font = {
            'family': 'Times New Roman',
            'weight': 'bold',
            'size': 20
        }
        x = pd.date_range('2015-7-5', periods=len(pred), freq='5d')
        plt.plot(x, original_well, color='red', label='Real  GWL', linewidth=3.0)
        plt.plot(x, pred_well, color='blue', label='Predicted  GWL', linewidth=3.0)
        plt.title('GCN_LSTM Well-{} Cij'.format(i + 1), fontdict=font)
        plt.xlabel('Time', fontdict=font)
        plt.ylabel('GWL', fontdict=font)
        plt.xticks(rotation=45, weight='bold', size=18)
        plt.yticks(weight='bold', size=18)
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.legend(loc='upper left', prop={'size': 18})
        # plt.show()
        plt.savefig(
            './PNG/GCN-LSTM-Well-{}.png'.format(
                i + 1))
        # Output evaluation metrics
        outdata = ['Well-{}'.format(i + 1), r2, mse, Rmse, mae, rrmse]
        outDataArr.append(outdata)

        # Total well
        # Output evaluation metrics
        pred_list = pred_well.tolist()
        original_list = original_well.tolist()

        # total well
        for j in range(len(pred_list)):
            outPredictionData = pred_list[j]
            outOriginalData = original_list[j]
            totalPrediction.append(outPredictionData)
            totalOriginal.append(outOriginalData)
            Abs = abs(pred_list[j] - original_list[j])
            outSinglePredictionData = [pred_list[j], original_list[j], Abs]
            outPredictionArr.append(outSinglePredictionData)

    # totalOriginal type--  <class 'list'>
    print("\ntotalOriginal type-- ", type(totalOriginal))
    print("\ntotalOriginal-- ", totalOriginal)
    Original = DataFrame(totalOriginal)
    Prediction = DataFrame(totalPrediction)
    print("Original", Original)
    print("Original length", len(Original))
    print("totalOriginal", totalOriginal)
    print("Prediction", Prediction)
    print("totalPrediction", totalPrediction)
    # total_r2 = r2_score(totalPrediction, totalOriginal)
    total_r2 = r2_score(totalOriginal, totalPrediction)
    total_mse = mean_squared_error(totalPrediction, totalOriginal)
    total_Rmse = rmse(totalPrediction, totalOriginal)
    total_mae = mean_absolute_error(totalPrediction, totalOriginal)
    total_rRMSE = rRMSE(Original, Prediction)
    print("Total Index--")
    print("r2", total_r2)
    print("mse", total_mse)
    print("rmse", total_Rmse)
    print("mae", total_mae)
    print("rRMSE", total_rRMSE)
    # Output evaluation metrics
    outIndexdata = [total_r2, total_mse, total_Rmse, total_mae, total_rRMSE]
    outTotalIndexArr.append(outIndexdata)

    # Save evaluation metrics to a file
    outDataArr = pd.DataFrame(outDataArr)
    outDataArr.rename(columns={0: 'well', 1: 'R2', 2: 'mse', 3: 'rmse', 4: 'mae', 5: 'rRMSE'}, inplace=True)
    print(outDataArr)
    outDataArr.to_csv(outIndex)

    # Save prediction to a file
    # Total well
    outPredictionArr = pd.DataFrame(outPredictionArr)

    outPredictionArr.rename(columns={0: 'WellPrediction', 1: 'WellOriginal', 2: 'WellAbs',
                                     }, inplace=True)

    # Save prediction to a file

    print(outPredictionArr)
    outPredictionArr.to_csv(outPrediction)

    # Save evaluation metrics to a file
    outTotalIndexArr = pd.DataFrame(outTotalIndexArr)
    outTotalIndexArr.rename(columns={0: 'R2', 1: 'mse', 2: 'rmse', 3: 'mae', 4: 'rRMSE'}, inplace=True)
    print(outTotalIndexArr)
    outTotalIndexArr.to_csv(outTotalIndex)
