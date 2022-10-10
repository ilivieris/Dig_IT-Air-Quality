import pandas as pd
import numpy  as np
from   tqdm   import tqdm 
from tsmoothie.smoother import *


def create_dataset(df = None, Lag = 1, Horizon = 1, targetSeries = None, overlap = 1):
    
    if (targetSeries is None):
        targetSeries = df.columns[-1]
    
    dataX, dataY, dataDate, dataXTime, dataYTime = [], [], [], [], []
    
    for i in tqdm( range(0, df.shape[0] + 1  - Lag - Horizon, overlap) ):
        
        dataX.append( df.to_numpy()[i:(i+Lag)] )        
        dataY.append( df[ targetSeries ].to_numpy()[i + Lag : i + Lag + Horizon] )
        dataDate.append( df.index[i + Lag : i + Lag + Horizon].tolist() )

        
    return ( np.array(dataX).astype(np.float32), 
             np.array(dataY).astype(np.float32), 
             np.array(dataDate) )


class DataAugmentation():
    '''
    Data Augmentation using Smoothing (ConvolutionSmoother)
    '''
    
    def __init__(self, df, args):
        self._df  = df
        self._args = args
        print('[INFO] Data Augmentation using Smoothing (ConvolutionSmoother) is setup')
        
    def createInstances(self, window_len, window_type):
        '''
        Create new instances (X, Y)
        '''
        
        # Setup Smoother
        #
        smoother = ConvolutionSmoother(window_len  = window_len,
                                       window_type = window_type)
        
        # Apply smoother
        #
        df_temp = self._df.copy()
        #
        smoother.smooth( df_temp[ self._args.targetSeries ] )
        #
        df_temp[ self._args.targetSeries ] = smoother.smooth_data.T

        X, Y, _      = create_dataset(df           = df_temp, 
                                      Lag          = self._args.Lag, 
                                      Horizon      = self._args.Horizon, 
                                      targetSeries = self._args.targetSeries,
                                      overlap      = self._args.Horizon,)
        
        return X, Y