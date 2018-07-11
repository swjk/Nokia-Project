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
                plt.savefig("BaselineGraph/" + column)

def main():
    dV1 = DataVisualisation("ExampleDailycharts.xlsx", "Data")
    dV1.readDataSheet()
    dV1.boxPlot()



if __name__== "__main__":
  main()
