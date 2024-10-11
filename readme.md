This is a TensorFlow implementation of GCN-LSTM: A Spatiotemporal Prediction Model for Groundwater Level.

##Requirements:

    1.tensorflow
    
    2.scipy
    
    3.sklearn
    
    4.keras
    
    5.numpy
    
    6.matplotlib
    
    7.pandas
    
    8.math
    
    9.xlrd
    
    10.xlwt
    
##Directory description:

    Data: Includes input data for all scripts.
    
    GCN-LSTM-TensorFlow: py file that contains all models and methods
    
    Models: Includes saved, trained models.
    
##Data Description:

    well-all-Original: The groundwater level observation data of 16 wells in Zhangjiajie every five days from 2003 to 2017.
    
    Attribute_Cij: The attribute similarity matrix is calculated based on the properties of distance, elevation, slope and aspect between 16 wells.
    
    Spatial_adj_tin: This is a spatial weighting matrix calculated from the structure of 16-well-graph created by the Delaunay triangulation.
    
##Run the demo:

    Our baselines included:
    
    1.GCN_LSTM
    
    2.LSTM_Dropout_FC
    
    3.LSTM_FC
    
    4.LSTM_Dropout_LSTM
    
    5.LSTM_LSTM
        
    The python implementations of  models were in the baselines.py; The GCN Layer were in spektral_gcn.py and spektral_utilities.py.
    
    The method.py includes the methods for calculating adjacency matrices and other general methods.
        
##Code call relationship

    ![avatar]('https://note.youdao.com/s/c3K6YmoY')


 
    

    
