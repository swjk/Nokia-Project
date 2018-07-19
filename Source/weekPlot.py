import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Figure(object):
    @staticmethod
    def saveFigure(fig,w_inches, h_inches, location):
        fig.set_size_inches(w_inches, h_inches)
        plt.savefig(location, dpi=100)

class Writer(object):
    @staticmethod
    #Data can be both Dataframe or Series
    def writeExcel(file,sheet, data):
        writer = pd.ExcelWriter(file)
        data.to_excel(writer,sheet)
        writer.save()


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
        groupedRow = self.dataFrame.groupby(self.dataFrame[self.dataFrame.columns[0]].dt.weekday_name, sort=False)
        print("Plotting")
        for index,column in enumerate(self.dataFrame):
            if(index >= 2):
                fig = plt.figure(   )
                plt.title(column)
                newdataFrame = pd.DataFrame()
                days = []
                for name, group in groupedRow:
                    newdataFrame[name] = group[column].reset_index(drop=True)
                    days.append(name)
                newdataFrame.boxplot()
                plt.xticks([1, 2, 3, 4, 5, 6, 7], days)
                Figure.saveFigure(fig,10,6,"BoxPlots/" + column)

    def meanAndErrorPlot(self):
        groupedRow = self.dataFrame.groupby(self.dataFrame[self.dataFrame.columns[0]].dt.weekday_name, sort=False)
        print("Plotting")
        for index,column in enumerate(self.dataFrame):
            if(index >= 2):
                fig = plt.figure()
                plt.title(column)
                newdataFrame = pd.DataFrame()
                weekday = 0
                days = []
                for name, group in groupedRow:
                    weekday += 1
                    newdataFrame[name] = group[column].reset_index(drop=True)
                    mean     = newdataFrame[name].mean()
                    std = newdataFrame[name].std()
                    days.append(name)
                    plt.errorbar(weekday,mean,std,fmt='o')
                plt.xticks([1, 2, 3, 4, 5, 6, 7], days)
                Figure.saveFigure(fig,10,6,"ErrorBarPlots/" + column)

    def kpiWeeklyPlot(self):
        groupedRow = self.dataFrame.groupby(self.dataFrame[self.dataFrame.columns[0]].dt.weekday_name, sort=False)
        print("Plotting")
        for index,column in enumerate(self.dataFrame):
            if(index >= 2):
                fig = plt.figure()
                plt.title(column)
                for name,group in groupedRow:
                    group = group.reset_index(drop=True)
                    plt.plot(group[column], label = name)
                    plt.legend(loc='upper left')
                    plt.xticks([0,1, 2, 3, 4, 5, 6], ["Week1","Week2","Week3","Week4","Week5","Week6","Week7"])
                Figure.saveFigure(fig,10,6, "GroupKPI/" + column)

    def correlationPlot(self):
        self.dataFrame = self.dataFrame.drop("Period start time", axis=1)
        self.dataFrame = self.dataFrame.drop("PLMN Name", axis=1)

        correlations = self.dataFrame.corr()
        correlations = correlations.applymap(lambda x: x if abs(x) > 0.95 else 0)
        #print (correlations)
        newdataFrame = pd.DataFrame()

        for name, row in correlations.iterrows():
            newdata = []
            row = row[row != 0]
            rowindex = row.index
            rowindex = rowindex.drop(name, errors='ignore')
            if(not rowindex.empty):
                fig = plt.figure()
                plt.title(name)
                plt.plot(self.dataFrame[name], label = name)
                newdata.append(name)
                for index in rowindex:
                    plt.plot(self.dataFrame[index], label = index)
                    plt.legend(loc ='upper left')
                    newdata.append(index)
                newdataFrame = newdataFrame.append(pd.Series(newdata, name = name))
                Figure.saveFigure(fig,10,6,"../Correlation/ThresholdGraph/" + name)

        print (newdataFrame)
        Writer.writeExcel("../Correlation/ThresholdGraph/ThresholdGraphExcel.xlsx","Sheet1", newdataFrame)



        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        fig.colorbar(cax)
        Figure.saveFigure(fig,18,10,"../Correlation/KPIS")

    def correlation(self):
        groupedRow = self.dataFrame.groupby(self.dataFrame[self.dataFrame.columns[0]].dt.weekday_name, sort=False)
        dayCorrelation = np.zeros((7,7))
        daysGroup1 = []
        group1c = 0
        for name1, group1 in groupedRow:
            daysGroup1.append(name1)
            group1 = group1.drop("Period start time", axis=1)
            group1 = group1.drop("PLMN Name", axis=1)
            newdataFrame1 = pd.DataFrame()
            newdataFrame1 = group1.reset_index(drop=True)
            group2c = 0
            for name2, group2 in groupedRow:
                group2 = group2.drop("Period start time", axis=1)
                group2 = group2.drop("PLMN Name", axis=1)
                newdataFrame2 = pd.DataFrame()
                newdataFrame2 = group2.reset_index(drop=True)
                colCorrelation = newdataFrame1.corrwith(newdataFrame2)
                averageBetweenTwoDays = colCorrelation.mean()
                dayCorrelation[group1c][group2c] =  averageBetweenTwoDays
                group2c += 1
            group1c += 1
        correlationFrame = pd.DataFrame(dayCorrelation)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlationFrame, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ax.set_xticklabels(['']+daysGroup1)
        ax.set_yticklabels(['']+daysGroup1)
        Figure.saveFigure(fig,18,10,"Correlation/Days")


    def weekcorrelation(self):
        groupedRow = self.dataFrame.groupby(self.dataFrame[self.dataFrame.columns[0]].dt.weekday_name, sort=False)
        for name, group in groupedRow:
            group = group.drop("Period start time", axis=1)
            group = group.drop("PLMN Name", axis=1)
            groupdataFrame = pd.DataFrame()
            groupdataFrame = group.reset_index(drop=True)
            groupdataFrameTranspose = groupdataFrame.T
            correlation = groupdataFrameTranspose.corr()
            print(correlation)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(correlation, vmin=0.9999, vmax=1)
            fig.colorbar(cax)
            ax.set_xticklabels(['']+["Week1","Week2","Week3","Week4","Week5","Week6","Week7"])
            ax.set_yticklabels(['']+["Week1","Week2","Week3","Week4","Week5","Week6","Week7"])
            Figure.saveFigure(fig,18,10,"Correlation/DayWeek" + name)



def main():
    dV1 = DataVisualisation("../Data/WeeklyData.xlsx", "Data")
    dV1.readDataSheet()
    #dV1.meanAndErrorPlot()
    #dV1.boxPlot()x
    dV1.correlationPlot()
    #dV1.correlation()
    #dV1.weekcorrelation()
    #dV1.kpiWeeklyPlot()

if __name__== "__main__":
  main()
