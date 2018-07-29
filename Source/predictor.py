import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMAResults
from sklearn import datasets, linear_model
from scipy.optimize import brute



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
            dailyGroupedData = DataManipulation.groupDataByDay(self.dataFrame)
            firstDay         = next(iter(dailyGroupedData.groups))
            trainingData     = pd.DataFrame()
            forcastData      = pd.DataFrame()

            for day, dayReadings in dailyGroupedData:
                if day <= firstDay + trainingDays:
                    trainingData = trainingData.append(dayReadings)
                else:
                    forcastData = forcastData.append(dayReadings)

            return (trainingData, forcastData)

class DataManipulation(object):
    @staticmethod
    def groupDataByDay(data):
        return data.groupby(data.index.dayofyear, sort=False)
    @staticmethod
    def groupDataByDayName(data):
            return data.groupby(data.index.weekday, sort=False)


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

"""
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
ARIMA
"""
def analyseARIMA(trainingData):
    targetColumn = trainingData['Avg act UEs DL']
    lag_acf  = acf(targetColumn,  nlags=170)
    lag_pacf = pacf(targetColumn, nlags=170, method='ols')

    figure = plt.figure()
    plt.subplot(121)
    plt.stem(lag_acf)
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=-1.96/np.sqrt(len(targetColumn)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(targetColumn)), linestyle='--', color='gray')
    plt.xlabel('Lag')
    plt.ylabel('ACF')


    plt.subplot(122)
    plt.stem(lag_pacf)
    plt.axhline(y=0, linestyle='-', color='black')
    plt.axhline(y=-1.96/np.sqrt(len(targetColumn)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(targetColumn)), linestyle='--', color='gray')
    plt.xlabel('Lag')
    plt.ylabel('PACF')

    plt.tight_layout()
    plt.show()

def arimaModel(order,day1KPIData):
    print(order)
    try:
        model       = ARIMA(day1KPIData, order=order)
        model1Result= model.fit(disp=0)
        residuals = pd.DataFrame(model1Result.resid)
        print(residuals)
        if residuals.isnull().values.any():
            print(10**10)
            return 10**10
        else:
            print("Std")
            print(residuals.std())
            return residuals.std()
    except KeyboardInterrupt:
        return
    except:
        return 10**10



def analyseARIMA2(trainingData, futureData, predictionHours):
    #Assuming prediction hours <= 24
    groupedByDayName = DataManipulation.groupDataByDayName(trainingData)
    firstDateTime = pd.to_datetime(trainingData.index.values[-1]) + pd.Timedelta(hours=1)
    timeRangeDay1 = pd.date_range(start=(firstDateTime),
                              end=pd.Timestamp(year=firstDateTime.year, month=firstDateTime.month, day=firstDateTime.day, hour=23),
                              freq='H')
    timeRangeDay2 = pd.date_range(start=pd.Timestamp(year=firstDateTime.year, month=firstDateTime.month, day=firstDateTime.day) + pd.DateOffset(1), periods=predictionHours-len(timeRangeDay1), freq='H')

    forcastLengthDay1 = len(timeRangeDay1)
    forcastLengthDay2 = len(timeRangeDay2)

    day1Name = timeRangeDay1.weekday[0]
    day2Name = (day1Name + 1) % 7

    day1Data = groupedByDayName.get_group(day1Name)
    day2Data = groupedByDayName.get_group(day2Name)

    for kpi in day1Data:
        day1KPIData = day1Data[kpi]
        day1KPIDataReset = day1KPIData.reset_index(drop=True)
        day2KPIData = day2Data[kpi]
        day2KPIDataReset = day1KPIData.reset_index(drop=True)

        try:
            grid = (slice(1,8,1), slice(0,3,1), slice(0,3,1))
            optimizeOrderPara = brute(arimaModel, grid, args=[(day1KPIDataReset)], finish=None)
            print("OPTIMISED")
            print(optimizeOrderPara)
            model1 = ARIMA(day1KPIDataReset, order=(int(optimizeOrderPara[0]),int(optimizeOrderPara[1]),int(optimizeOrderPara[2]))).fit(disp=0)

        except:
            continue

        predictionDataDay1 = model1.forecast(forcastLengthDay1)
        predictionDataDay2 = None
        if forcastLengthDay2 > 0:
            try:
                grid = (slice(1,8,1), slice(0,3,1), slice(0,3,1))
                optimizeOrderPara = brute(arimaModel, grid, args=[(day1KPIDataReset)], finish=None)
                print("OPTIMISED")
                print(optimizeOrderPara)
                model2 = ARIMA(day1KPIDataReset, order=(int(optimizeOrderPara[0]),int(optimizeOrderPara[1]),int(optimizeOrderPara[2]))).fit(disp=0)

            except:
                continue

            predictionDataDay2 = model2.forecast(forcastLengthDay2)

        predictionDataFrame = pd.DataFrame()
        if predictionDataDay2 == None:
            forcastData = predictionDataDay1[0]
            predictionDataFrame[kpi] = forcastData
            predictionDataFrame.set_index(timeRangeDay1, inplace=True)
        else:
            forcastData = predictionDataDay1[0] + predictionDataDay2[0]
            predictionDataFrame[kpi] = forcastData
            predictionDataFrame.set_index(timeRangeDay1.append(timeRangeDay2), inplace =True)
        GraphData.comparisonPlot([{'data': predictionDataFrame, 'name': 'prediction'},
                                  {'data':futureData, 'name': 'actual'}], '../New/')



def test(trainingData):
    targetColumn = trainingData['Avg act UEs DL']
    targetData = pd.concat([targetColumn, targetColumn],ignore_index=True)
    #model = ARIMA(targetColumn, order=(1,0,1))
    #results_ARIMA = model.fit()
    #print(results_ARIMA.aic)



"""
def seasonalARIMA(trainingData, futureData):
    targetColumn = trainingData['Avg act UEs DL']
    targetData = pd.concat([targetColumn, targetColumn],ignore_index=True)



    #model = SARIMAX(targetData, order= (2,1,1), seasonal_order=(1,0,1,168), enforce_stationarity=False, enforce_invertibility=False)
    #model_fit = model.fit(disp=False)

    futureForcast = 24
    predictionData = model_fit.forecast(futureForcast)

    fig = plt.figure()
    plt.plot(predictionData)
    plt.plot(futureData['Avg act UEs DL'])
    plt.show()
    print (predictionData)

"""

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

    #correlationDailyData  = correlation(spreadSheet2In, spreadSheet2Out, cfg.dailyThreshold)
    #correlationHourlyData = correlation(spreadSheet1In, spreadSheet3Out, cfg.hourlyThreshold)

    #linearRegressionAlgorithm(trainingData, correlationHourlyData)
    #analyseARIMA(trainingData)
    #test(trainingData)
    #seasonalARIMA(trainingData,futureData)
    analyseARIMA2(trainingData,futureData, 24)


    #predictionData = referenceAlgorithm(spreadSheet1In, cfg.referenceWindowSize)
    #GraphData.comparisonPlot([spreadSheet1In.dataFrame, predictionData], cfg.referenceGraphLocation)




    #maeFrame = calculateMAE(spreadSheet1In.dataFrame, predictionData)
    #spreadSheet1Out.writeExcelData(maeFrame)



if __name__== "__main__":
  main()
