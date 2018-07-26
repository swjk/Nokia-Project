import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from statsmodels.tsa.arima_model import ARIMA
from sklearn import datasets, linear_model

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
            #Drop rows with all NaN
            self.dataFrame.dropna(axis=0, how='all', inplace=True)
            #Make sure columns are numeric
            self.dataFrame = self.dataFrame.apply(pd.to_numeric, errors='ignore')



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
        def comparisonPlot(dataDicList, storeLocation):
            for columnLabel, _ in dataDicList[0]['data'].iteritems():
                fig = GraphData.createFigure(columnLabel)
                for dic in dataDicList:
                    plt.plot(dic['data'][columnLabel], label = dic['name'] )
                    plt.legend(loc='upper left')
                GraphData.saveFigure(fig,10,6, storeLocation + columnLabel)
        @staticmethod
        def formPlotDictionary(name, data):
            return {'name': name, 'data': data}


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

def calculateMeanAverage2(time, trainingData):
    prevDayTime  = time - pd.Timedelta(days=1)
    prevWeekTime = time - pd.Timedelta(days=7)

    prevDaySeries  = trainingData.loc[prevDayTime]
    prevWeekSeries = trainingData.loc[prevWeekTime]

    meanTimeForcast      = prevDaySeries.add(prevWeekSeries).divide(2)
    meanTimeForcast.name = time

    return meanTimeForcast

def referenceAlgorithm2(trainingData):
    #Forcast Next Day
    predictionData = pd.DataFrame()
    timeRange = pd.date_range(start=(trainingData.index.values[-1] + pd.Timedelta(hours=1)), periods=24, freq='H')
    for time in timeRange:
        predictionData = predictionData.append(calculateMeanAverage2(time, trainingData))
    predictionData.index.name = 'Period start time'
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



def correlation(spreadSheetIn, spreadSheetOut, upperThreshold):
    #Correlation is always absolute value
    correlations = spreadSheetIn.dataFrame.corr()
    correlations = correlations.applymap(lambda x: abs(x) if abs(x) > upperThreshold else 0)
    correlationResult = pd.DataFrame()

    for kpiName, kpiCorrelations in correlations.iterrows():
        kpiCorrelations = kpiCorrelations[kpiCorrelations != 0]
        kpiCorrelations = kpiCorrelations[kpiCorrelations != 1]
        correlationResult = correlationResult.append(pd.Series(kpiCorrelations, name= kpiName))

    spreadSheetOut.writeExcelData(correlationResult)
    return correlationResult

"""
-------------------------------------------------------------------------------
"""


"""
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
Regression Algorithm
"""

def linearRegressionAlgorithm(trainingData, correlationHourlyData):
    #E.g. Avg act UEs DL
    correlationSeries = correlationHourlyData['Avg act UEs DL']
    topCorrelation = correlationSeries.sort_values(ascending=False)
    topCorrelation.dropna(inplace=True)
    subset = trainingData[topCorrelation.index.values].values

    targetColumn = trainingData['Avg act UEs DL']
    targetColumn = targetColumn.values

    regr = linear_model.LinearRegression()
    regr.fit(subset, targetColumn)


    print (subset)
    print (topCorrelation)


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
        elif type == 'corrHourly':
            return ExportData(cfg.writeCorrelationHourlyLocation, cfg.writeCorrelationHourlySheetName)
        elif type == 'corrDaily':
            return ExportData(cfg.writeCorrelationDailyLocation, cfg.writeCorrelationSDailySheetName)

def forcastReference(trainingData, futureData):
    predictionData = referenceAlgorithm2(trainingData)
    GraphData.comparisonPlot([formPlotDictionary("prediction", predictionData),
                              formPlotDictionary("actual", futureData.head(len(predictionData)))],cfg.referenceGraphLocation)


def main():
    spreadSheet1In  = importDataFromExcel('hourly')
    spreadSheet1Out = exportDataToExcel('stats')
    spreadSheet3Out = exportDataToExcel('corrHourly')

    spreadSheet2In  = importDataFromExcel('daily')
    spreadSheet2Out = exportDataToExcel('corrDaily')

    trainingData, futureData = spreadSheet1In.dataSeparation(cfg.trainingDays)

    #forcastReference(trainingData, futureData)

    correlationDailyData  = correlation(spreadSheet2In, spreadSheet2Out, cfg.dailyThreshold)
    correlationHourlyData = correlation(spreadSheet1In, spreadSheet3Out, cfg.hourlyThreshold)

    linearRegressionAlgorithm(trainingData, correlationHourlyData)
    #arimaAlgorithm(spreadSheet1In)




    #predictionData = referenceAlgorithm(spreadSheet1In, cfg.referenceWindowSize)
    #GraphData.comparisonPlot([spreadSheet1In.dataFrame, predictionData], cfg.referenceGraphLocation)




    #maeFrame = calculateMAE(spreadSheet1In.dataFrame, predictionData)
    #spreadSheet1Out.writeExcelData(maeFrame)



if __name__== "__main__":
  main()
