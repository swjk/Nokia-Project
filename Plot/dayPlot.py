import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Figure(object):
    @staticmethod
    def saveFigure(fig,w_inches, h_inches, location):
        fig.set_size_inches(w_inches, h_inches)
        plt.savefig(location, dpi=100)

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


def findTrafficKPIs(fileName, sheet):
    dV = DataVisualisation(fileName, sheet)
    dV.readDataSheet()
    secondColumn = dV.dataFrame[dV.dataFrame.columns[2]]
    dataReady    = secondColumn.drop(secondColumn.index[:72])
    dataReady = dataReady.reset_index(drop=True)
    print (dataReady.rolling(4))
    plt.plot(dataReady)
    plt.show()
    print (dataReady.kurtosis())







def main():
    findTrafficKPIs("../Data/HourData.xlsx", "SIGN_1")
    findTrafficKPIs("../Data/HourData.xlsx", "HO_1")

if __name__== "__main__":
  main()
