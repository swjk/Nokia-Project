import keras
from keras.layers import LSTM,Dense, Dropout
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

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

    def parseDate(self):
        if 'Period start time' in self.dataFrame:
            self.dataFrame['Period start time'] = pd.to_datetime(self.dataFrame['Period start time'], format="%d/%m/%Y %H:%M:%S", utc=True, errors='coerce')

    def removeEmptyColumns(self):
        self.dataFrame = self.dataFrame.dropna(axis = 'columns', how = 'all')

    def saveAsCsv(self, fileName):
        self.dataFrame.to_csv(fileName)

    def removeNaRows(self, data = None):
        if data is None:
            self.dataFrame = self.dataFrame.dropna(axis = 'index', how = 'any' )
        else:
            data.dropna(axis = 'index', how = 'any', inplace = True)

    def series_to_supervised(self, data):
        #data is list of list
        n_vars = data.shape[1]
        cols,names   = list(), list()

        df     = pd.DataFrame(data)
        cols.append(df)
        cols.append(df.ix[:,0].shift(-1))
        agg = pd.concat(cols, axis=1)
        names = np.arange(14)
        agg.columns = names
        agg.dropna(inplace=True)
        print (agg)

        return agg







    def LSTM(self):
        self.dataFrame.drop('PLMN Name', axis = 1, inplace = True)
        self.dataFrame = self.dataFrame.set_index('Period start time')

        #Get columns from start:stop:step
        subsetData = self.dataFrame.loc[:,'Avg IP thp DL QCI8':'ERAB_ADD_SETUP_ATT_QCI8 (M8006C204)']
        self.removeNaRows(subsetData)

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(subsetData.values[1:])

        supervisedData = self.series_to_supervised(scaled)
        supervisedValue = supervisedData.values
        n_train_hours = 24 * 6

        train = supervisedValue[:n_train_hours, :]
        test  = supervisedValue[n_train_hours:, :]
        train_X, train_Y = train[:, :-1], train[:, -1]
        test_X, test_Y = test[:,:-1], test[:,-1]

        #Reshape to be 3d [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)



        model = keras.Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        outlayer = Dense(1)
        model.add(outlayer)
        #model.add(Dropout(0.5))

        model.compile(loss='mae', optimizer='adam')
        history = model.fit(train_X, train_Y, epochs=50, validation_data=(test_X, test_Y ), verbose = 2, shuffle=False)

        print (outlayer.get_weights())

        #plt.plot(history.history['loss'], label='train')
        #plt.plot(history.history['val_loss'], label='test')
        #plt.legend()
        #plt.show()



        #Making predictions
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

        inv_yhat = np.concatenate((yhat, test_X[:,1:]), axis = 1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]

        test_Y = test_Y.reshape((len(test_Y), 1))
        inv_y = np.concatenate((test_Y, test_X[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        fig = plt.figure()
        plt.plot(inv_y)
        plt.plot(inv_yhat)
        Figure.saveFigure(fig,18,10,"ForecastTest")

        #print (test_X)








def main():
    dV1 = DataVisualisation("../Data/HourlyData.xlsx", "reportRaw1")
    dV1.readDataSheet()
    dV1.parseDate()
    dV1.removeEmptyColumns()
    #dV1.saveAsCsv("dailydata.csv")
    dV1.LSTM()

if __name__== "__main__":
  main()
