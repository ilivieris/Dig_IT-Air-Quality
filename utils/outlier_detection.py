# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Built-in library
#
import numpy  as np

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Visualization library
#
import matplotlib.pyplot as plt


def detect_outliers(A, Method = 'IQR', verbose = False, figsize = (20, 3)):

    Array = A[ ~np.isnan(A) ]
    if (Method == 'IQR'):
        # 1st quartile (25%)
        Q1 = np.percentile(Array, 25)

        # 3rd quartile (75%)
        Q3 = np.percentile(Array, 75)

        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Bounds
        Lower_bound = Q1 - outlier_step
        Upper_bound = Q3 + outlier_step
        
    elif (Method == 'STD'):
        Lower_bound = np.mean(Array) - 3.0*np.std(Array)
        Upper_bound = np.mean(Array) + 3.0*np.std(Array)
        
    else:
        print('[ERROR] Method: {} not known'.format(Method))
        return
    
    if (verbose):
        print('Lower bound: %.3f' % Lower_bound)
        print('Upper bound: %.3f' % Upper_bound)

        plt.figure( figsize = figsize )
        plt.plot( A )
        
        plt.axhline(xmin=0, xmax=A.shape[0], y=Lower_bound, color = 'r', linestyle = '--' )
        plt.axhline(xmin=0, xmax=A.shape[0], y=Upper_bound, color = 'r', linestyle = '--' )
        
        plt.show()
            
    # Determine a list of indices of outliers for feature col
    #
    A = np.where( A < Lower_bound, np.NaN, A)
    A = np.where( A > Upper_bound, np.NaN, A)

    return A