import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config as cfg

class Correlation(object):
    @staticmethod
    def correlation(dataIn, threshold):
        """
        Pairwise correlation between columns with values kept if absolute value higher than threshold

        Parameters
        ----------
        dataIn : pandas dataframe
                Data that pairwise correlation done on
        threshold : float
                Value which correlation must be above in order to be kept

        Returns
        -------
        pandas dataframe
                Same sized dataframe as dataIn, with correlation values above threshold

        """
        correlations = dataIn.corr()
        correlations = correlations.applymap(lambda x: abs(x) if abs(x) > threshold else 0)
        correlationResult = pd.DataFrame()

        for kpiName, kpiCorrelations in correlations.iterrows():
            kpiCorrelations = kpiCorrelations[kpiCorrelations != 0]
            kpiCorrelations = kpiCorrelations[kpiCorrelations != 1]
            correlationResult = correlationResult.append(pd.Series(kpiCorrelations, name= kpiName))

        return correlationResult
