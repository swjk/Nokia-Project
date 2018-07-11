import tensorflow as tf
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt




def readFile():
    fileName = "ExampleDailycharts.xlsx"
    dataFrame= pd.read_excel(fileName, "Data")
    groupedRow = dataFrame.groupby(dataFrame[dataFrame.columns[0]].dt.weekday_name)
    #ayseries= dataFrame[dataFrame.columns[0]].dt.weekday_name
    #print (dayseries.groupby(lambda x: x))

    #for key, item in groupedRow:
    #    print (groupedRow.get_group(key))
    #groupedRow.plot("Monday","E-UTRAN Init E-RAB acc (LTE_5060I)")
    #plt.show()

    for index,column in enumerate(dataFrame):
        if(index >= 2):
            plt.figure()
            plt.title(column)
            for name, group in groupedRow:
                group[column].plot()
                #subdataFrame = pd.concat([group['Period start time'],group[column]],axis=1)
                #print (subdataFrame)
                #subdataFrame.plot()
                #plt.xaxis.set_major_formatter(plt.dates.DateFormatter('%Y-%m-%d'))

            plt.savefig("BaselineGraph/" + column)


def main():
  print("Hello World!")
  hello = tf.constant('Hello')
  sess = tf.Session()
  print(sess.run(hello))


if __name__== "__main__":
  main()
  readFile()
