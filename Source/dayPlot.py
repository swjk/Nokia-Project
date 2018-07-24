import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, special
import numpy as np
from sklearn.neighbors import NearestNeighbors

class Figure(object):
    @staticmethod
    def saveFigure(fig,w_inches, h_inches, location):
        fig.set_size_inches(w_inches, h_inches)
        plt.savefig(location, dpi=100, format='png')



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

    def parseDate(self):
        if 'Period start time' in self.dataFrame:
            self.dataFrame['Period start time'] = pd.to_datetime(self.dataFrame['Period start time'], format="%d/%m/%Y %H:%M:%S", utc=True, errors='coerce')

    def correlationPlot(self):
        self.dataFrame = self.dataFrame.set_index('Period start time')
        print (self.dataFrame)
        daily_grouped_dataFrame = self.dataFrame.groupby(self.dataFrame.index.dayofyear, sort=False)
        for name, values in self.dataFrame.iteritems():

            if not (name in ['Period start time', 'PLMN Name']):
                fig = plt.figure()
                plt.title(name + ' ( ' + self.dataFrame[name].iloc[0] + ')')

                for (daynum, day_readings) in daily_grouped_dataFrame:
                    if not (pd.isnull(day_readings[name]).all()):
                        plt.plot(day_readings[name])
                        #day_readings[name] = day_readings[name].astype(float)

                        #upperlimit = 3 * day_readings[name].std() + day_readings[name].mean()
                        #lowerlimit = -3 * day_readings[name].std() + day_readings[name].mean()

                        #print (day_readings[name])
                        #print(day_readings[name].shape)
                        #day_readings_temp = day_readings.copy()
                        #day_readings_temp[name] = upperlimit
                        #plt.plot(day_readings_temp[name])
                        #day_readings_temp[name] = day_readings_temp[name].map(lambda x : upperlimit)
                        #print (day_readings_temp[name])
                        #day_readings_temp[name] = upperlimit
                        #plt.plot(day_readings_temp[name],'r--','xkcd:maroon')
                        #day_readings_temp[name] = lowerlimit
                        #plt.plot(day_readings_temp[name],'r--', 'xkcd:maroon')
                self.dataFrame[name] = self.dataFrame[name].iloc[1:]
                self.dataFrame[name] = self.dataFrame[name].astype(float)
                autocorrelation = self.dataFrame[name].autocorr(24)
                print (autocorrelation)
                ax = plt.gca()
                plt.text(0.5, 0.5, "Autocorrelation: " + "{:.2f}".format(autocorrelation) , horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)


                Figure.saveFigure(fig,10,6,"../DailyPlot/" + name)



    def mutualInformation(self, k_neighbours):
        series1 = self.dataFrame['E-UTRAN Init E-RAB acc'].dropna()
        series2 = self.dataFrame['E-RAB DR RAN'].dropna()
        series1 = series1.iloc[1:]
        series2 = series2.iloc[1:]

        arrayTest1 = (series1.values)
        arrayTest2 = series2.values

        arrayCom = [[arrayV,arrayTest2[count]] for count,arrayV in enumerate(arrayTest1)]
        print (arrayCom)


        arrayTest1 = arrayTest1.reshape(-1, 1)
        arrayTest2 = arrayTest2.reshape(-1,1)

        nbrsSer1 = NearestNeighbors().fit(arrayTest1)
        nbrsSer2 = NearestNeighbors().fit(arrayTest2)

        nbrsCom = NearestNeighbors(n_neighbors = k_neighbours, metric = mydist).fit(arrayCom)

        Ixy = special.digamma(k_neighbours)
        Ixytemp = 0
        IxyN = 0
        for indexCom , elemCom in enumerate(arrayCom):
            distance, indices = nbrsCom.kneighbors([elemCom])
            print (distance)
            halfe_i = np.sum(np.array(distance))
            print (halfe_i)
            dist1, ind1 = nbrsSer1.radius_neighbors(X = arrayTest1[indexCom].reshape(1, -1), radius = halfe_i)
            dist2, ind2 = nbrsSer2.radius_neighbors(X = arrayTest2[indexCom].reshape(1, -1), radius = halfe_i)
            IxyN += 1
            Ixytemp += special.digamma(len(ind1[0]) + 1 ) * special.digamma(len(ind2[0]) + 1 )

        Ixy -= (1 / IxyN) * Ixytemp
        Ixy += special.digamma(IxyN)

        print(Ixy)




        #Test
        samples = [[1,4], [0,10], [5,7], [3,45]]
        test = NearestNeighbors(n_neighbors = 1, metric = mydist).fit(samples)
        print(test.kneighbors())





def mydist(point1,point2):
    return max(abs(point1[0] - point2[0]), abs(point1[1] - point2[1]))




def main():
    dV1 = DataVisualisation("../Data/HourlyData.xlsx", "reportRaw1")
    dV1.readDataSheet()
    dV1.parseDate()
    #dV1.correlationPlot()
    dV1.mutualInformation(4)

if __name__== "__main__":
  main()
