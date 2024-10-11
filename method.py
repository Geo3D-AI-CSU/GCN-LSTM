import numpy as np
from sklearn.metrics import mean_squared_error
import xlrd
import pandas as pd
import os
import haversine

def createXY(dataset, n_past):
    """
    Create input sequences and corresponding target values for time series forecasting.

    Parameters:
    - dataset (numpy.ndarray): The input dataset, which contains historical time series data.
    - n_past (int): The length of the input sequences, i.e., how many past time steps to use for prediction.

    Returns:
    - dataX (numpy.ndarray): Input sequences with a shape of (number of sequences, n_past, number of features).
    - dataY (numpy.ndarray): Corresponding target values for each input sequence.
    """
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    print(len(dataX))
    return np.array(dataX), np.array(dataY)


def rmse(y1, y2):
    """
    Calculate the Root Mean Squared Error (RMSE) between two sets of values.

    Parameters:
    - y1 (array-like): The first set of values.
    - y2 (array-like): The second set of values.

    Returns:
    - rmse (float): The Root Mean Squared Error between y1 and y2.
    """
    return np.sqrt(mean_squared_error(y1, y2))


def calSpatialAdj(x_arr, y_arr):
    """
    Calculate a spatial adjacency matrix based on well locations.

    Parameters:
    - x_arr (array-like): An array of x-coordinates (e.g., longitudes) for wells.
    - y_arr (array-like): An array of y-coordinates (e.g., latitudes) for wells.

    Returns:
    - adj (pandas.DataFrame): The spatial adjacency matrix where each element represents the proximity between wells.
    """
    adj_Rowarr = []
    for i in range(len(x_arr)):
        adj_Colarr = []
        for j in range(len(x_arr)):
            if i != j:
                well_i = (y_arr[i], x_arr[i])
                well_j = (y_arr[j], x_arr[j])
                print(well_i, well_j)
                distance = haversine.haversine(well_i, well_j)
                inverse_distance = 1 / (distance**2 + 1)
                print(distance)
            else:
                inverse_distance = 1
            adj_Colarr.append(inverse_distance)
        adj_Rowarr.append(adj_Colarr)
    adj = pd.DataFrame(adj_Rowarr)
    return adj


def calAttribute_Cij(a1, b1, c1, d1, a2, b2, c2, d2):
    """
    Calculate the cosine similarity between two attribute vectors.

    Parameters:
    - a1 (float): The first component of the first attribute vector.
    - b1 (float): The second component of the first attribute vector.
    - c1 (float): The third component of the first attribute vector.
    - d1 (float): The fourth component of the first attribute vector.
    - a2 (float): The first component of the second attribute vector.
    - b2 (float): The second component of the second attribute vector.
    - c2 (float): The third component of the second attribute vector.
    - d2 (float): The fourth component of the second attribute vector.

    Returns:
    - cos (float): The cosine similarity between the two attribute vectors.
    """
    m = a1 * a2 + b1 * b2 + c1 * c2 + d1 * d2
    n = pow(a1**2 + b1**2 + c1**2 + d1**2, 0.5) * pow(a2**2 + b2**2 + c2**2 + d2**2, 0.5)
    cos = m / n
    return cos


if __name__ == '__main__':
    df = pd.read_excel(r'./data/WellXY.xls', sheet_name='Sheet1')
    print(df.iloc[:, 0])
    WellX_arr = np.array(df.iloc[:, 0])
    WellY_arr = np.array(df.iloc[:, 1])
    print(len(WellX_arr), len(WellY_arr))
    adj = calSpatialAdj(WellX_arr, WellY_arr)
    writer = pd.ExcelWriter(r'./data/Spatial_adj.xls')
    adj.to_excel(writer, 'sheet_1', float_format='%.5f')
    writer.save()
    writer.close()

    data1 = pd.read_excel(r'./data/Attribute_Well.xls', index_col='num', sheet_name='Sheet2')
    data1 = data1.to_numpy()
    print(data1)
    Cij = np.zeros((16, 16))
    for i in range(16):
        for j in range(0, 16):
            if i != j:
                a1 = data1[i][0]
                b1 = data1[i][1]
                c1 = data1[i][2]
                d1 = data1[i][3]
                a2 = data1[j][0]
                b2 = data1[j][1]
                c2 = data1[j][2]
                d2 = data1[j][3]
                Cij[i][j] = calAttribute_Cij(a1, b1, c1, d1, a2, b2, c2, d2)
                Cij[j][i] = calAttribute_Cij(a1, b1, c1, d1, a2, b2, c2, d2)
            else:
                Cij[i][j] = 1
    print(Cij)
    data = pd.DataFrame(Cij)
    writer = pd.ExcelWriter(r'./data/Attribute_Cij.xls')
    data.to_excel(writer, 'sheet_1', float_format='%.5f')
    writer.save()
    writer.close()
