import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

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

        #Read Excel Data Sheet And Store in DataFrame
        def readDataSheet(self):
            self.dataFrame = pd.read_excel(self.fileName, self.sheet)
            return self.dataFrame

        #Parse Time Data As DateTime Object
        def parseDate(self):
            if 'Period start time' in self.dataFrame:
                self.dataFrame['Period start time'] = pd.to_datetime(self.dataFrame['Period start time'], format="%d/%m/%Y %H:%M:%S", utc=True, errors='coerce')

        def groupDataByDay(self):
            self.dataFrame = self.dataFrame.set_index('Period start time')
            return self.dataFrame.groupby(self.dataFrame.index.dayofyear, sort=False)



def calculateMeanAverage(windowData, windowSize, newData):
    windowData = windowData.append(newData)

    if (len(windowData.index) > windowSize):
        windowData = windowData.iloc[1:]
        return windowData, windowData.mean()
    return windowData, pd.Series()


def referenceAlgorithm(spreadSheet1In):
    windowData       = pd.DataFrame()
    forcastData      = pd.Series()
    maeData          = pd.Series()
    nData            = 0
    for index, hourReading in spreadSheet1In.dataFrame.iterrows():
        #forcastData is the prediction for this hourReading
        if not (forcastData.empty):
            if(maeData.empty):
                maeData = forcastData.subtract(hourReading).abs()
            else:
                maeData = maeData.add(forcastData.subtract(hourReading).abs())
            nData += 1
        windowData, forcastData = calculateMeanAverage(windowData, 4, hourReading)
    maeData = maeData.divide(nData)
    return maeData



def setDataUp(spreadSheet1In):
    spreadSheet1In.dataFrame.set_index('Period start time', inplace=True)
    spreadSheet1In.dataFrame.drop('PLMN Name', axis=1, inplace=True)
    spreadSheet1In.dataFrame.dropna(axis=0, how='all', inplace=True)

    #Drop rows with index NaT
    spreadSheet1In.dataFrame = spreadSheet1In.dataFrame.loc[pd.notnull(spreadSheet1In.dataFrame.index)]


    '''
    dailyGroupedData = spreadSheet1In.groupDataByDay()
    firstDay = next(iter(dailyGroupedData.groups))

    trainingData = pd.DataFrame()
    forcastData = pd.DspreadSheet1OutataFrame()

    for day, dayReadings in dailyGroupedData:
        if day <= firstDay + trainingDays:
            trainingData = trainingData.append(dayReadings)
        else:
            forcastData = forcastData.append(dayReadings)

    forcastData.dropna(axis=0, how='all', inplace=True)
    trainingData.dropna(axis=0, how='all', inplace=True)

    return (trainingData, forcastData)
    '''


def ImportDataFromExcel():
        return ImportData("../Data/HourlyData.xlsx", "reportRaw1")

def ExportDataToExcel():
        return ExportData("../Data/ReferenceData.xlsx", "sheet1")


def main():
    spreadSheet1In  = ImportDataFromExcel()
    spreadSheet1Out = ExportDataToExcel()
    setDataUp(spreadSheet1In)
    maeDataRA = referenceAlgorithm(spreadSheet1In)
    spreadSheet1Out.writeExcelData(maeDataRA)




if __name__== "__main__":
  main()
