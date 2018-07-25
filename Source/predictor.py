import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from statsmodels.tsa.arima_model import ARIMA


class ExportData(object):
    fileName = ""
    sheet    = ""

    def __init__(self,fileName,sheet):
        self.fileName = fileName
        self.sheet    = sheet

    def writeExcelData(self, data):
        writer = pd.ExcelWriter(self.fileName)
        data.to_excel(writer,self.sheet)
        writer.save()


class ImportData(object):
        fileName    = ""
        sheet       = ""
        dataFrame   = pd.DataFrame()

        def __init__(self,fileName,sheet):
            self.fileName = fileName
            self.sheet    = sheet
            self.readDataSheet()
            self.parseDate()
            self.setDataUp()

        #Read Excel Data Sheet And Store in DataFrame
        def readDataSheet(self):
            self.dataFrame = pd.read_excel(self.fileName, self.sheet)
            return self.dataFrame

        #Parse Time Data As DateTime Object
        def parseDate(self):
            if 'Period start time' in self.dataFrame:
                self.dataFrame['Period start time'] = pd.to_datetime(self.dataFrame['Period start time'], format="%d/%m/%Y %H:%M:%S", utc=True, errors='coerce')

        def groupDataByDay(self):
            return self.dataFrame.groupby(self.dataFrame.index.dayofyear, sort=False)

        def setDataUp(self):
            #Set datetime to be index
            self.dataFrame.set_index('Period start time', inplace=True)
            #Drop PLMN tag
            self.dataFrame.drop('PLMN Name', axis=1, inplace=True)
            #Drop all columns more than 90% empty
            self.dataFrame.dropna(axis=1, thresh=0.9*len(self.dataFrame), inplace=True)
            #Drop rows with index NaT
            self.dataFrame = self.dataFrame.loc[pd.notnull(self.dataFrame.index)]


        def dataSeparation(self, trainingDays):
            dailyGroupedData = self.groupDataByDay()
            firstDay         = next(iter(dailyGroupedData.groups))
            trainingData     = pd.DataFrame()
            forcastData      = pd.DataFrame()

            for day, dayReadings in dailyGroupedData:
                if day <= firstDay + trainingDays:
                    trainingData = trainingData.append(dayReadings)
                else:
                    forcastData = forcastData.append(dayReadings)

            return (trainingData, forcastData)

class GraphData(object):
        @staticmethod
        def saveFigure(fig,w_inches, h_inches, location):
            fig.set_size_inches(w_inches, h_inches)
            plt.savefig(location, dpi=100, format='png')

        @staticmethod
        def createFigure(title):
            fig = plt.figure()
            plt.title(title)
            return fig

        @staticmethod
        def comparisonPlot(dataList, storeLocation):
            for columnLabel, _ in dataList[0].iteritems():
                fig = GraphData.createFigure(columnLabel)
                for data in dataList:
                    plt.plot(data[columnLabel])
                GraphData.saveFigure(fig,10,6, storeLocation + columnLabel)


"""
-------------------------------------------------------------------------------
Reference Algorithm
"""

def calculateMeanAverage(windowData,windowSize, newData):
    windowData = windowData.append(newData)
    if (len(windowData.index) > windowSize):
        windowData = windowData.iloc[1:]
        return windowData, windowData.mean()
    return windowData, pd.Series()

def referenceAlgorithm(spreadSheetIn, windowSize):
    predictionData = pd.DataFrame()
    windowData     = pd.DataFrame()
    forcastSeries  = pd.Series()

    for index, hourReading in spreadSheetIn.dataFrame.iterrows():
        forcastSeries.name = index
        predictionData = predictionData.append(forcastSeries)
        windowData, forcastSeries = calculateMeanAverage(windowData, windowSize, hourReading)
    return predictionData

"""
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
First Correlation Algorithm
"""
def arimaAlgorithm(spreadSheetIn):
    firstcolumn = pd.Series(spreadSheetIn.dataFrame.iloc[:,0])
    train = firstcolumn[:20]
    test = firstcolumn[20:]
    model = ARIMA(train, order = (2,12,0))
    model_fit = model.fit(disp= 0)
    print(model_fit.summary())



def correlationDaily(spreadSheetIn, spreadSheetOut, upperThreshold):
    correlations = spreadSheetIn.dataFrame.corr()
    correlations = correlations.applymap(lambda x: x if abs(x) > upperThreshold else 0)
    correlationResult = pd.DataFrame()

    for kpiName, kpiCorrelations in correlations.iterrows():
        kpiCorrelations = kpiCorrelations[kpiCorrelations != 0]
        correlationResult = correlationResult.append(pd.Series(kpiCorrelations, name= kpiName))

    spreadSheetOut.writeExcelData(correlationResult)
    return correlationResult



def correlation1Algorithm(correlationResult):
    for kpiName, kpiCorrelations in correlationResult.iterrows():
        kpiCorrelations.dropna(inplace=True)
        print (kpiCorrelations)


    #for index, hourReading in spreadSheetIn.dataFrame.iterrows():




"""
-------------------------------------------------------------------------------
"""



def calculateMAE(originalData, predictionData):
    maeFrame = pd.DataFrame()
    for columnLabel, _ in originalData.iteritems():
        comparisonFrame = pd.concat([originalData[columnLabel],predictionData[columnLabel]], axis=1)
        comparisonFrame.dropna(axis=0, how='any', inplace=True)
        mae = pd.Series(comparisonFrame.iloc[:,0]).subtract(pd.Series(comparisonFrame.iloc[:,1])).abs().mean()
        maeFrame[columnLabel] = pd.Series(mae, index=["MAE"])
    return maeFrame

def importDataFromExcel(type):
        if type == 'hourly':
            return ImportData(cfg.hourlyDataLocation, cfg.hourlySheetName)
        elif type == 'daily':
            return ImportData(cfg.dailyDataLocation, cfg.dailySheetName)

def exportDataToExcel(type):
        if type == 'stats':
            return ExportData(cfg.writeNewDataLocation, cfg.writeNewDataSheetName)
        elif type == 'corr':
            return ExportData(cfg.writeCorrelationLocation, cfg.writeCorrelationSheetName)


def main():
    spreadSheet1In  = importDataFromExcel('hourly')
    spreadSheet2In  = importDataFromExcel('daily')
    spreadSheet1Out = exportDataToExcel('stats')
    spreadSheet2Out = exportDataToExcel('corr')

    trainingData, forcastData = spreadSheet1In.dataSeparation(14)

    arimaAlgorithm(spreadSheet1In)

    #correlationDailyData = correlationDaily(spreadSheet2In, spreadSheet2Out, cfg.upperThreshold)
    #correlation1Algorithm(correlationDailyData)

    #predictionData = referenceAlgorithm(spreadSheet1In, cfg.referenceWindowSize)
    #GraphData.comparisonPlot([spreadSheet1In.dataFrame, predictionData], cfg.referenceGraphLocation)




    #maeFrame = calculateMAE(spreadSheet1In.dataFrame, predictionData)
    #spreadSheet1Out.writeExcelData(maeFrame)



if __name__== "__main__":
  main()
