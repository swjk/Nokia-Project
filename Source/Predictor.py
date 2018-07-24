import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config as cfg

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


def calculateMeanAverage(windowData,windowSize, newData):
    windowData = windowData.append(newData)
    if (len(windowData.index) > windowSize):
        windowData = windowData.iloc[1:]
        return windowData, windowData.mean()
    return windowData, pd.Series()

def referenceAlgorithm(spreadSheet1In):
    predictionData = pd.DataFrame()
    windowData     = pd.DataFrame()
    forcastSeries  = pd.Series()

    for index, hourReading in spreadSheet1In.dataFrame.iterrows():
        forcastSeries.name = index
        predictionData = predictionData.append(forcastSeries)
        windowData, forcastSeries = calculateMeanAverage(windowData, 4, hourReading)
    return predictionData


def setDataUp(spreadSheetIn):
    #Set datetime to be index
    spreadSheetIn.dataFrame.set_index('Period start time', inplace=True)
    #Drop PLMN tag
    spreadSheetIn.dataFrame.drop('PLMN Name', axis=1, inplace=True)
    #Drop all columns more than 90% empty
    spreadSheetIn.dataFrame=spreadSheetIn.dataFrame.replace({'':np.nan})
    spreadSheetIn.dataFrame.dropna(axis=1, thresh=0.9*len(spreadSheetIn.dataFrame), inplace=True)
    #Drop rows with index NaT
    spreadSheetIn.dataFrame = spreadSheetIn.dataFrame.loc[pd.notnull(spreadSheetIn.dataFrame.index)]



def ImportDataFromExcel():
        return ImportData("../Data/HourlyData.xlsx", "reportRaw1")

def ExportDataToExcel():
        return ExportData("../Data/ReferenceData.xlsx", "sheet1")


def main():
    spreadSheet1In  = ImportDataFromExcel()
    spreadSheet1Out = ExportDataToExcel()

    setDataUp(spreadSheet1In)

    predictionData = referenceAlgorithm(spreadSheet1In)
    GraphData.comparisonPlot([spreadSheet1In.dataFrame, predictionData], cfg.referenceGraphLocation)
    #spreadSheet1Out.writeExcelData(predictionData)

if __name__== "__main__":
  main()
