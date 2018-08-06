import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
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


import store as store

"""
-------------------------------------------------------------------------------
Reference Algorithm
"""
def calculateMeanAverage(time, trainingData):
    """
    Work out mean average between series day before and series week before

    Parameters
    ----------
    time: datetime
        time point we want to forecast

    trainingData : pandas dataframe
        training data containing one or more KPIs

    Returns
    -------
    pandas dataframe
        predicted data
    """
    prevDayTime  = time - pd.Timedelta(days=1)
    prevWeekTime = time - pd.Timedelta(days=7)

    #if (prevDayTime.dayofweek == 4 or prevDayTime.dayofweek == 6):
    #    prevDayTime = time -pd.Timedelta(days=6)


    prevDaySeries  = trainingData.loc[prevDayTime]
    prevWeekSeries = trainingData.loc[prevWeekTime]

    meanTimeForcast      = prevDaySeries.add(prevWeekSeries).divide(2)
    meanTimeForcast.name = time

    return meanTimeForcast

def referenceAlgorithm(trainingData):
    """
    Prediction of future data time points using average of time point
    from previous day and previous week

    Parameters
    ----------
    trainingData : pandas dataframe
        training data containing one or more KPIs

    Returns
    -------
    pandas dataframe
        predicted data
    """
    predictionData = pd.DataFrame()
    print(trainingData)
    timeRange = pd.date_range(start=(trainingData.index.values[-1] + pd.Timedelta(hours=1)), periods=cfg.predictionHours, freq='H')
    for time in timeRange:
        predictionData = predictionData.append(calculateMeanAverage(time, trainingData))
    predictionData.index.name = 'Period start time'
    return predictionData

def forcastReference(trainingData, futureData):
    predictionResult = referenceAlgorithm(trainingData)
    #GraphData.comparisonDataFramePlot([GraphData.formPlotDictionary("prediction", predictionResult),
    #                          GraphData.formPlotDictionary("actual", futureData.head(len(predictionResult)))],cfg.referenceGraphLocation)
    return predictionResult

"""
-------------------------------------------------------------------------------
Regression Algorithm
"""
def linearRegressionAlgorithm(trainingData, futureData, correlationHourlyData, predictionType):
    """
    Algorithm used to create forcast for one or more KPIs in training data.
    For each KPI, will first find dependent KPIs and model these using predictionType.
    Will then use linear regression to combine these to predict future for this KPI

    Will ignore KPI will no dependencies

    Parameters
    ----------
    trainingData : pandas dataframe
        training data containing one or more KPIs
    futureData: pandas dataframe
        data containing one or more KPIs to be predicted
    correlationHourlyData: pandas dataframe
        correlation information for all KPIs above defined threshold
    predictionType: string
        "ARIMA" or "LSTM" - used to model dependencies

    Returns
    -------

    """
    predictionResult = pd.DataFrame()
    for kpi in trainingData:
        print (kpi)
        if kpi not in correlationHourlyData.columns:
            print("KPI has no dependencies")
        else:
            correlationSeries = correlationHourlyData[kpi]
            topCorrelationKPIs = correlationSeries.sort_values(ascending=False)
            topCorrelationKPIs.dropna(inplace=True)
            topCorrelationKPIs = topCorrelationKPIs.head(cfg.topNCorrelations)

            topCorrelationKPIsNames  = topCorrelationKPIs.index.values
            topCorrelationKPIsValues = trainingData[topCorrelationKPIs.index.values].values

            targetKPI = trainingData[kpi]
            targetKPI = targetKPI.values

            regr = linear_model.LinearRegression()
            regr.fit(topCorrelationKPIsValues, targetKPI)


            if predictionType == "ARIMA":
                graphStoreLocation = cfg.lrARIMAGraphLocation
                forcastTopCorrelationKPIsValues = predictionARIMA(trainingData[topCorrelationKPIsNames],
                                                    futureData[topCorrelationKPIsNames],
                                                    cfg.predictionHours)
            elif predictionType == "LSTM":
                graphStoreLocation = cfg.lrLSTMGraphLocation
                forcastTopCorrelationKPIsValues = predictionLSTM(trainingData[topCorrelationKPIsNames],
                                                    futureData[topCorrelationKPIsNames],
                                                    cfg.predictionHours)

            forcastTargetKPIValues = regr.predict(forcastTopCorrelationKPIsValues.values)
            predictionResult[kpi] = forcastTargetKPIValues
            predictionResult.set_index(forcastTopCorrelationKPIsValues.index, inplace=True)

            predictionStdUpperLimit = pd.DataFrame(index = forcastTopCorrelationKPIsValues.index)
            predictionStdLowerLimit = pd.DataFrame(index = forcastTopCorrelationKPIsValues.index)
            predictionShiftLimit = forcastLimits(trainingData,cfg.predictionHours,kpi).multiply(cfg.forcastLimitMultiplicationConstant)

            predictionStdUpperLimit[kpi] = predictionResult[kpi].add(predictionShiftLimit)
            predictionStdLowerLimit[kpi] = predictionResult[kpi].subtract(predictionShiftLimit)

            #GraphData.comparisonSeriesPlot([{'data': predictionResult[kpi], 'dependencies': topCorrelationKPIsNames, 'name': 'prediction'},
            #                                {'data': predictionStdUpperLimit, 'style': '--', 'name': "upperlimit"},
            #                                {'data': predictionStdLowerLimit, 'style': '--', 'name': "lowerlimit"},
            #                                {'data':futureData[kpi], 'name': 'actual'}], graphStoreLocation)
    return predictionResult

#------Forcast Limits--------------
def forcastLimits(trainingData, predictionHours, kpi):
    """
    Calculate average standard deviation of weekly time points (e.g Mon 9am).

    Parameters
    ----------
    trainingData : pandas dataframe
        training data
    predictionHours: int
        number of hours predicted
    kpi: string
        kpi that calculation should be carried out on

    Returns
    -------
    pandas series
        series containing calculation for each time point
    """
    groupedByDayName = DataManipulation.groupDataByDayName(trainingData)
    firstDateTime = pd.to_datetime(trainingData.index.values[-1]) + pd.Timedelta(hours=1)
    if (firstDateTime + pd.Timedelta(hours=predictionHours)).date() == firstDateTime.date():
        dateTimeRangeDay1 = pd.date_range(start=(firstDateTime), end = firstDateTime + pd.Timedelta(hours=predictionHours), freq='H')
    else:
        dateTimeRangeDay1 = pd.date_range(start=(firstDateTime),
                              end=pd.Timestamp(year=firstDateTime.year, month=firstDateTime.month, day=firstDateTime.day, hour=23),
                              freq='H')
    dateTimeRangeDay2 = pd.date_range(start=pd.Timestamp(year=firstDateTime.year, month=firstDateTime.month, day=firstDateTime.day) + pd.DateOffset(1), periods=predictionHours-len(dateTimeRangeDay1), freq='H')

    forcastLengthDay1 = len(dateTimeRangeDay1)
    forcastLengthDay2 = len(dateTimeRangeDay2)

    day1Name = dateTimeRangeDay1.weekday[0]
    day2Name = (day1Name + 1) % 7

    day1Data = groupedByDayName.get_group(day1Name)
    day2Data = groupedByDayName.get_group(day2Name)

    day1KPIData = day1Data[kpi]
    day2KPIData = day2Data[kpi]

    if forcastLengthDay2 > 0:
        day1KPIRollingStd = day1KPIData.rolling(cfg.forcastLimitWindowSize, min_periods=1, center=True).std()
        day2KPIRollingStd = day2KPIData.rolling(cfg.forcastLimitWindowSize, min_periods=1, center=True).std()

        day1GroupedByTime = DataManipulation.groupDataByTime(day1KPIRollingStd)
        day2GroupedByTime = DataManipulation.groupDataByTime(day2KPIRollingStd)

        day1KPIAvgRollingStd = dateTimeRangeDay1.to_series().map(lambda dateTime: day1GroupedByTime.get_group(dateTime.time()).mean())
        day2KPIAvgRollingStd = dateTimeRangeDay2.to_series().map(lambda dateTime: day2GroupedByTime.get_group(dateTime.time()).mean())

        return day1KPIAvgRollingStd.append(day2KPIAvgRollingStd)
    else:
        day1KPIRollingStd = day1KPIData.rolling(cfg.forcastLimitWindowSize, min_periods=1, center=True).std()
        day1GroupedByTime = DataManipulation.groupDataByTime(day1KPIRollingStd)
        day1KPIAvgRollingStd = dateTimeRangeDay1.to_series().map(lambda dateTime: day1GroupedByTime.get_group(dateTime.time()).mean())
        return day1KPIAvgRollingStd


"""
-------------------------------------------------------------------------------
ARIMA
"""
def analyseARIMA(trainingData, kpi):
    """
    Analyse training Data to show autocorrelation and partial autocorrelation
    """
    targetColumn = trainingData[kpi]
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

def arimaModel(order,trainingData):
    """
    Create ARIMA model

    Parameters
    ----------
    order : tuple(p,d,q)
        The parameters for the arima model
    trainingData: pandas series
        Training data for one kpi

    Returns
    -------
    int
        If no exception - residual standard deviation, how well model fits training data
        Else - 10**10 (large number)
    """
    print(order)
    try:
        model       = ARIMA(trainingData, order=order)
        model1Result= model.fit(disp=0)
        residuals = pd.DataFrame(model1Result.resid)
        if residuals.isnull().values.any():
            return 10**10
        else:
            return residuals.std()
    except KeyboardInterrupt:
        return
    except:
        return 10**10


def predictionARIMA(trainingData, futureData, predictionHours):
    """
    Predict future data based on training data. Uses standard arima with grid search
    of parameters to find optimum to model per kpi how it progresses over time

    Parameters
    ----------
    trainingData : pandas dataframe
        contains data for one or more KPIs. Index must be a DateTimeIndex
    futureData: pandas dataframe
        contains data for one or more KPIs which should be forecasted. Index must be a DateTimeIndex
    predictionHours: int
        number of hours to predict

    Returns
    -------
    pandas dataframe
        prediction outcome of one or more KPIs forcastedd. Index must be a DateTimeIndex
    """
    groupedByDayName = DataManipulation.groupDataByDayName(trainingData)
    firstDateTime = pd.to_datetime(trainingData.index.values[-1]) + pd.Timedelta(hours=1)

    if (firstDateTime + pd.Timedelta(hours=predictionHours-1)).date() == firstDateTime.date():
        dateTimeRangeDay1 = pd.date_range(start=(firstDateTime), end = firstDateTime + pd.Timedelta(hours=predictionHours-1), freq='H')
    else:
        dateTimeRangeDay1 = pd.date_range(start=(firstDateTime),
                              end=pd.Timestamp(year=firstDateTime.year, month=firstDateTime.month, day=firstDateTime.day, hour=23),
                              freq='H')
    dateTimeRangeDay2 = pd.date_range(start=pd.Timestamp(year=firstDateTime.year, month=firstDateTime.month, day=firstDateTime.day) + pd.DateOffset(1), periods=(predictionHours)-len(dateTimeRangeDay1), freq='H')

    forcastLengthDay1 = len(dateTimeRangeDay1)
    forcastLengthDay2 = len(dateTimeRangeDay2)

    day1Name = dateTimeRangeDay1.weekday[0]
    day2Name = (day1Name + 1) % 7

    day1Data = groupedByDayName.get_group(day1Name)
    day2Data = groupedByDayName.get_group(day2Name)

    predictionResult = pd.DataFrame()
    for kpi in day1Data:
        day1KPIData = day1Data[kpi]
        day1KPIDataReset = day1KPIData.reset_index(drop=True)
        day2KPIData = day2Data[kpi]
        day2KPIDataReset = day1KPIData.reset_index(drop=True)

        try:
            grid = (slice(1,8,1), slice(0,3,1), slice(0,3,1))
            optimizeOrderPara = brute(arimaModel, grid, args=[(day1KPIDataReset)], finish=None)
            model1 = ARIMA(day1KPIDataReset, order=(int(optimizeOrderPara[0]),int(optimizeOrderPara[1]),int(optimizeOrderPara[2]))).fit(disp=0)

        except:
            continue
        print("Model1 -Optimised")
        predictionDataDay1 = model1.forecast(forcastLengthDay1)
        predictionDataDay2 = None
        if forcastLengthDay2 > 0:
            try:
                grid = (slice(1,8,1), slice(0,3,1), slice(0,3,1))
                optimizeOrderPara = brute(arimaModel, grid, args=[(day1KPIDataReset)], finish=None)
                model2 = ARIMA(day1KPIDataReset, order=(int(optimizeOrderPara[0]),int(optimizeOrderPara[1]),int(optimizeOrderPara[2]))).fit(disp=0)

            except:
                continue
            print("Model2 -Optimised")
            predictionDataDay2 = model2.forecast(forcastLengthDay2)

        if predictionDataDay2 == None:
            forcastData = predictionDataDay1[0]
            predictionResult[kpi] = forcastData
            predictionResult.set_index(dateTimeRangeDay1, inplace=True)
        else:
            print(predictionDataDay1[0])
            print(predictionDataDay2[0])
            forecastData = np.concatenate([predictionDataDay1[0], predictionDataDay2[0]])
            print (forecastData)
            predictionResult[kpi] = forecastData
            predictionResult.set_index(dateTimeRangeDay1.append(dateTimeRangeDay2), inplace =True)
        #GraphData.comparisonSeriesPlot([{'data': predictionResult[kpi],  'name': 'prediction'},
        #                              {'data':futureData[kpi], 'name': 'actual'}], cfg.arimaGraphLocation)
    return predictionResult

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


"""
-------------------------------------------------------------------------------
LSTM Neural Network
"""
def fitLSTM(trainInput, trainOutput, nneurons, nbatch, nepoch):
    """
    Create LSTM model

    Parameters
    ----------
    trainInput : np array
        model training input
    trainOuput:  np array
        model training output used in backpropagation to alter weights
    nneurons: int
        neuron number in hidden layer
    nbatch: int
        batch number
    nepoch: int
        epoch number

    Returns
    -------
    Sequential object
        fitted model to training data
    """
    X = trainInput.reshape(trainInput.shape[0], 1, trainInput.shape[1])

    model = Sequential()
    model.add(LSTM(nneurons, batch_input_shape=(nbatch, X.shape[1], X.shape[2])))
    model.add(Dense(trainOutput.shape[1]))
    model.compile(loss="mean_squared_error", optimizer='adam')

    for i in range (nepoch):
        model.reset_states()
        model.fit(X,trainOutput,epochs=1, batch_size=nbatch, verbose=0, shuffle=False)
    return model

def predictionLSTM(trainingData,futureData, predictionHours):
    """
    Predict future data based on training data. Makes use of long short
    term memory neural network per kpi to model how it progresses over time

    Parameters
    ----------
    trainingData : pandas dataframe
        contains data for one or more KPIs. Index must be a DateTimeIndex
    futureData: pandas dataframe
        contains data for one or more KPIs which should be forecasted. Index must be a DateTimeIndex
    predictionHours: int
        number of hours to predict

    Returns
    -------
    pandas dataframe
        prediction outcome of one or more KPIs forcastedd. Index must be a DateTimeIndex
    """

    # TODO: MAY NEED TO CHECK IF STATIONARY AND THEN INVERT AT END

    removeFirstNRows = len(trainingData) % predictionHours
    trainingData     = trainingData.iloc[removeFirstNRows:]

    firstDateTime = pd.to_datetime(trainingData.index.values[-1]) + pd.Timedelta(hours=1)
    dateTimeRange = pd.date_range(start=(firstDateTime), end = firstDateTime + pd.Timedelta(hours=predictionHours-1), freq='H')

    predictionResult = pd.DataFrame()
    for kpi in trainingData:
        print(kpi)
        kpiSeries = trainingData[kpi]
        # index, datetime, value
        kpiFrame  = kpiSeries.reset_index()
        groupedByPredHourMultiple  = DataManipulation.groupDataByPredictHourMultiple(kpiFrame)

        #groupLabels days(Mon,Tue..) only if each group 0:00 - 23:00
        groupLabels = []
        groupValues = []

        for group in groupedByPredHourMultiple:
            groupLabel = group[1].iloc[0,0].weekday() / 3 - 1
            groupLabels.append(groupLabel)
            groupValue = group[1].iloc[:,1].values
            groupValues.append(groupValue)

        scaler = MinMaxScaler(feature_range=(-1,1))
        scaledGroupValues = scaler.fit_transform(groupValues)

        groupLabels = np.array(groupLabels).reshape(-1,1)

        train       = np.concatenate((groupLabels, scaledGroupValues), axis=1)
        trainInput  = np.delete(train, len(train)-1, 0)
        trainOuput  = np.delete(train, 0, 0)
        trainForcastInput = train[-1]

        model    = fitLSTM(trainInput,trainOuput,30,1,30)
        forecast = model.predict(trainForcastInput.reshape(1,1,len(trainForcastInput)))

        forecastInvScale = np.delete(forecast[0],0,0)
        forecastInvScale = scaler.inverse_transform(forecastInvScale.reshape(1,-1))
        forecastInvScale = forecastInvScale[0,:]

        predictionResult[kpi] = pd.Series(forecastInvScale, index= dateTimeRange)

        #GraphData.comparisonSeriesPlot([{'data': predictionResult[kpi],  'name': 'prediction'},
        #                              {'data':futureData[kpi], 'name': 'actual'}], cfg.lstmGraphLocation)
    predictionResult.set_index(dateTimeRange, inplace=True)
    return predictionResult


def calculateMAE(originalData, predictionData):
    maeFrame = pd.DataFrame()
    for columnLabel, _ in originalData.iteritems():
        comparisonFrame = pd.concat([originalData[columnLabel],predictionData[columnLabel]], axis=1)
        comparisonFrame.dropna(axis=0, how='any', inplace=True)
        mae = pd.Series(comparisonFrame.iloc[:,0]).subtract(pd.Series(comparisonFrame.iloc[:,1])).abs().mean()
        maeFrame[columnLabel] = pd.Series(mae, index=["MAE"])
    return maeFrame

def calculateNMSE(futureData, predictionData, indexName):
    nmseFrame = pd.DataFrame()
    futureData.index = futureData.index.tz_localize(None)
    predictionData.index = predictionData.index.tz_localize(None)
    for columnLabel, _ in predictionData.iteritems():

        comparisonFrame = pd.concat([futureData[columnLabel], predictionData[columnLabel]], ignore_index=True, axis=1)
        comparisonFrame.dropna(axis=0, how='any', inplace=True)
        pmean = pd.Series(comparisonFrame.iloc[:,0]).mean()
        mmean = pd.Series(comparisonFrame.iloc[:,1]).mean()
        score = (((pd.Series(comparisonFrame.iloc[:,0]).subtract(pd.Series(comparisonFrame.iloc[:,1]))) ** 2) / ((pmean)*(mmean))).mean()
        print (score)
        nmseFrame[columnLabel] = pd.Series(abs(score), index =[indexName])
    print (nmseFrame)
    nmseFrame.replace([np.inf, -np.inf], np.nan, inplace=True)
    nmseFrame.dropna(axis=1, how='any', inplace=True)
    print(nmseFrame)
    print (nmseFrame.columns)
    print ("{} {}".format("Length of columns:", len(nmseFrame.columns)))
    return nmseFrame


def importDataFromExcel(type):
        """
        Import data from an excel spreadsheet

        Parameters
        ----------
        type : string
            determine whether data in hourly or daily

        Returns
        -------
        ImportData object
                object containing data in pandas dataframe
        """
        if type == 'hourly':
            return ImportData(cfg.hourlyDataLocation, cfg.hourlySheetName)
        elif type == 'daily':
            return ImportData(cfg.dailyDataLocation, cfg.dailySheetName)

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
    #import hourly data
    spreadSheet1In  = importDataFromExcel('hourly')
    #import daily data
    spreadSheet2In  = importDataFromExcel('daily')

    #export stats to rate models (e.g mae)
    spreadSheet1Out = ExportData(cfg.writeStatsDataLocation_Ref, cfg.writeStatsDataSheetName)
    spreadSheet2Out = ExportData(cfg.writeStatsDataLocation_S_LSTM, cfg.writeStatsDataSheetName)
    spreadSheet3Out = ExportData(cfg.writeStatsDataLocation_S_ARIMA, cfg.writeStatsDataSheetName)
    spreadSheet4Out = ExportData(cfg.writeStatsDataLocation_LR_LSTM, cfg.writeStatsDataSheetName)
    spreadSheet5Out = ExportData(cfg.writeStatsDataLocation_LR_ARIMA, cfg.writeStatsDataSheetName)
    spreadSheet6Out = ExportData(cfg.writeStatsNMSELocation, cfg.writeStatsNMSESheetName)
    spreadSheet7Out = ExportData(cfg.writeStatsNMSELocationAll, cfg.writeStatsNMSESheetName)
    #export hourly correlation
    #spreadSheet3Out = exportDataToExcel('corrHourly')

    #export daily correlation
    #spreadSheet2Out = exportDataToExcel('corrDaily')

    #split imported data into training and future data. future data is what is predicted
    trainingData, futureData = spreadSheet1In.dataSeparation(cfg.trainingDays)

    if sys.argv[1] == 'ref':
        #run reference algorithm
        print("------------Producing Reference Graphs")
        referencePredictionResult = forcastReference(trainingData, futureData)
        referenceNMSE = calculateNMSE(futureData, referencePredictionResult, "Reference")
        store.write(spreadSheet1Out, futureData, referencePredictionResult)
        finalResult = referenceNMSE
        spreadSheet6Out.writeExcelData(finalResult)
    elif sys.argv[1] == 'all':
        #run reference algorithm
        print("------------Producing Reference Graphs")
        referencePredictionResult = forcastReference(trainingData, futureData)
        referenceNMSE = calculateNMSE(futureData, referencePredictionResult, "Reference")
        store.write(spreadSheet1Out, futureData, referencePredictionResult)

        #run straight lstm neural network algorithm
        print("------------Producing Straight LSTM Graphs")
        straightLSTMPredictionResult = predictionLSTM(trainingData, futureData, cfg.predictionHours)
        straightLSTMNMSE = calculateNMSE(futureData, straightLSTMPredictionResult, "StraightLSTM")
        store.write(spreadSheet2Out, futureData, referencePredictionResult)

        #run straight arima algorithm
        print("------------Producing Straight ARIMA Graphs")
        straightARIMAPredictionResult = predictionARIMA(trainingData, futureData, cfg.predictionHours)
        straightARIMANMSE = calculateNMSE(futureData, straightARIMAPredictionResult, "StraightARIMA")
        store.write(spreadSheet3Out, futureData, referencePredictionResult)

        finalResult = pd.concat([referenceNMSE, straightLSTMNMSE, straightARIMANMSE])
        spreadSheet6Out.writeExcelData(finalResult)
        GraphData.comparisonSeriesPlotLog([{'data': finalResult.iloc[0,:],'name': 'reference', 'title': 'Comparison', 'rotate':90},
                                         {'data': finalResult.iloc[1,:],'name': 'straightLSTM'},
                                         {'data': finalResult.iloc[2,:],'name': 'straightARIMA'}],
                                          cfg.finalResultGraphLocation)


        #run linear regression lstm algorithm
        print("------------Producing Linear Regression LSTM Graphs")
        correlationHourlyData = Correlation.correlation(spreadSheet1In.dataFrame, cfg.hourlyCorrThreshold)
        lrLSTMPredictionResult = linearRegressionAlgorithm(trainingData, futureData, correlationHourlyData, "LSTM")
        lrLSTMNMSE = calculateNMSE(futureData, lrLSTMPredictionResult, "lrLSTM")
        store.write(spreadSheet4Out, futureData, referencePredictionResult)

        #run linear regression arima algorthim
        print("------------Producing Linear Regression ARIMA Graphs")
        correlationHourlyData = Correlation.correlation(spreadSheet1In.dataFrame, cfg.hourlyCorrThreshold)
        lrARIMAPredictionResult = linearRegressionAlgorithm(trainingData, futureData, correlationHourlyData, "ARIMA")
        lrARIMANMSE = calculateNMSE(futureData, lrARIMAPredictionResult, "lrARIMA")
        store.write(spreadSheet5Out, futureData, referencePredictionResult)

        finalResult = pd.concat([referenceNMSE,straightLSTMNMSE,straightARIMANMSE,lrLSTMNMSE,lrARIMANMSE])
        spreadSheet7Out.writeExcelData(finalResult)
        GraphData.comparisonSeriesPlotLog([{'data': finalResult.iloc[0,:],'name': 'reference', 'title': 'Comparison Graph', 'rotate':90},
                                         {'data': finalResult.iloc[1,:],'name': 'straightLSTM'},
                                         {'data': finalResult.iloc[2,:],'name': 'straightARIMA'},
                                         {'data': finalResult.iloc[3,:],'name': 'lrLSTM'},{'data': finalResult.iloc[4,:],'name': 'lrARIMA'}],
                                         cfg.finalResultGraphLocation)



    else:
        print ("Error - Argument must be provided")
        sys.exit()


if __name__== "__main__":
  main()
