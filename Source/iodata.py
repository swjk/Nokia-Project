import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config as cfg

from manipulatedata import DataManipulation

class ExportData(object):
    fileName = ""
    sheet    = ""

    def __init__(self,fileName,sheet):
        self.fileName = fileName
        self.sheet    = sheet

    def writeExcelData(self, data, startrow=0):
        writer = pd.ExcelWriter(self.fileName, engine='openpyxl')
        data.to_excel(writer,self.sheet, startrow=startrow)
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
            """
            Set up the dataframe such that it is clean to be used for further processing
            This involves setting index of dataframe to 'Period start time' column, removing PLMN column,
            dropping all columns and rows that are empty, and converting all entries into numerical form

            Parameters
            ----------
            Returns
            -------
            """
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
            """
            Used to separate the data into training data and future data, based on the size of training days given

            Parameters
            ----------
            trainingDays int
                The number of training data days.
            Returns
            -------
            2-tuple
                returns (trainingData, forecastData)
            """
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
