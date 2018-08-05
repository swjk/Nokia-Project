import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
import xlsxwriter
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMAResults
from sklearn import datasets, linear_model
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import brute
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from iodata import ExportData, ImportData
from graphdata import GraphData
from manipulatedata import DataManipulation
from correlation import Correlation

class DataObject(object):
    resultFrame = pd.DataFrame()

    def __init__(self,actualSeries, predictedSeries):
        predictedSeries = pd.Series(predictedSeries,name="Pred")
        actualSeries    = pd.Series(actualSeries,name="Act")

        self.resultFrame = self.resultFrame.append(predictedSeries)
        self.resultFrame =self.resultFrame.append(actualSeries)


    def getResultFrame(self):
        return self.resultFrame


def write(spreadSheetOut,futureData,predictionResult):
    writer = pd.ExcelWriter(spreadSheetOut.fileName, engine='xlsxwriter')
    start_row = 0
    workbook = writer.book
    sheetname1 = spreadSheetOut.sheet
    df1 = pd.DataFrame({'Data':[10,20,30]})

    df1.to_excel(writer, sheet_name = sheetname1, startrow = start_row)
    worksheet1 = writer.sheets[sheetname1]
    kpi_counter = 0

    for kpi in futureData:
        worksheet1.write(start_row,0,"KPI")
        worksheet1.write(start_row,1,kpi)
        start_row += 1
        testObject = DataObject(futureData[kpi], predictionResult[kpi])
        testObject.getResultFrame().to_excel(writer,   sheet_name = sheetname1 , startrow=start_row)
        start_row += 1
        start_row += 3
        kpi_counter += 1
        # if kpi_counter == 3:
        #    break
    workbook.close()

def exportDataToExcel(type):
        """
        Export data to an excel spreadsheet

        Parameters
        ----------
        type : string
            determine whether data exported in algorithm statistics(stats),
               or correlation (corrHourly, corrDaily)

        Returns
        -------
        ExportData object

        """
        if type == 'stats':
            return ExportData(cfg.writeStatsDataLocation, cfg.writeStatsDataSheetName)
        elif type == 'corrHourly':
            return ExportData(cfg.writeCorrelationHourlyLocation, cfg.writeCorrelationHourlySheetName)
        elif type == 'corrDaily':
            return ExportData(cfg.writeCorrelationDailyLocation, cfg.writeCorrelationSDailySheetName)


def main():
    #export stats to rate models (e.g mae)
    spreadSheet1Out = exportDataToExcel('stats')
    write(spreadSheet1Out)

if __name__== "__main__":
  main()
