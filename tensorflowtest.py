import tensorflow as tf
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt

class DataVisualisation(object):
    fileName    = ""
    sheet       = ""
    dataFrame   = pd.DataFrame()

    def __init__(self,fileName,sheet):
        self.fileName = fileName
        self.sheet    = sheet

    def readDataSheet(self):
        self.dataFrame = pd.read_excel(self.fileName, self.sheet)
        return self.dataFrame

    #Example BoxPlot
    def boxPlot(self):
        groupedRow = self.dataFrame.groupby(self.dataFrame[self.dataFrame.columns[0]].dt.weekday_name)
        print("Plotting")
        for index,column in enumerate(self.dataFrame):
            if(index >= 2):
                plt.figure()
                plt.title(column)
                newdataFrame = pd.DataFrame()
                for name, group in groupedRow:
                    newdataFrame[name] = group[column].reset_index(drop=True)

                newdataFrame.boxplot()
                plt.savefig("BoxPlots/" + column)

    def meanAndErrorPlot(self):
        groupedRow = self.dataFrame.groupby(self.dataFrame[self.dataFrame.columns[0]].dt.weekday_name)
        print("Plotting")
        for index,column in enumerate(self.dataFrame):
            if(index >= 2):
                plt.figure()
                plt.title(column)
                newdataFrame = pd.DataFrame()
                weekday = 0
                for name, group in groupedRow:
                    weekday += 1
                    newdataFrame[name] = group[column].reset_index(drop=True)
                    mean     = newdataFrame[name].mean()
                    std = newdataFrame[name].std()
                    plt.errorbar(weekday,mean,std,fmt='o')
                plt.savefig("ErrorBarPlots/" + column)


    def correlationPlot(self):
        self.dataFrame = self.dataFrame.drop("Period start time", axis=1)
        self.dataFrame = self.dataFrame.drop("PLMN Name", axis=1)
        correlations = self.dataFrame.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        plt.savefig("Correlation/correlationMatrix")





def main():
    dV1 = DataVisualisation("ExampleDailycharts.xlsx", "Data")
    dV1.readDataSheet()
    #dV1.meanAndErrorPlot()
    #dV1.boxPlot()
    dV1.correlationPlot()


if __name__== "__main__":
  main()
