import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
import os

class GraphData(object):
        @staticmethod
        def saveFigure(fig,w_inches, h_inches, location):
            """
            Save Figure in specified location
            """
            fig.set_size_inches(w_inches, h_inches)
            plt.savefig(location + '.png', dpi=100, format='png')

        @staticmethod
        def createFigure(title):
            """
            Create Figure with specified title
            """
            fig = plt.figure()
            plt.title(title)
            return fig

        @staticmethod
        def comparisonDataFramePlot(dataDicList, storeLocation):
            """
            For each column in the list of dataframes, draw comparison graph

            Parameters
            ----------
            dataDicList : list of pandas dataframe
                contains data to be graphed
            storeLocation: string
                location where graphs should be stored

            Returns
            -------
            """
            for columnLabel, _ in dataDicList[0]['data'].iteritems():
                fig = GraphData.createFigure(columnLabel)
                for dic in dataDicList:
                    if 'style' in dic:
                        plt.plot(dic['data'][columnLabel], dic['style'], label = dic['name'] )
                    else:
                        plt.plot(dic['data'][columnLabel], label = dic['name'] )

                    if 'dependencies' in dic:
                        plt.text(1, 1, "Dependent On: " + '\n'.join([str(i) for i in dic['dependencies']]), horizontalalignment='right',verticalalignment='top', transform=plt.gca().transAxes)
                plt.legend(loc='upper left')
                GraphData.saveFigure(fig,10,6, os.path.join(storeLocation,columnLabel))

        @staticmethod
        def comparisonSeriesPlot(dataDicList, storeLocation):
            """
            Draw one graph comparing the data in the list of Series

            Parameters
            ----------
            dataDicList : list of pandas series
                contains data to be graphed
            storeLocation: string
                location where graphs should be stored
            Returns
            -------
            """
            if 'title' in dataDicList[0]:
                name = dataDicList[0]['title']
            else:
                name = dataDicList[0]['data'].name

            fig = GraphData.createFigure(name)
            for dic in dataDicList:
                if 'style' in dic:
                        plt.plot(dic['data'], dic['style'], label=dic['name'])
                else:
                        plt.plot(dic['data'], label = dic['name'])
                if 'dependencies' in dic:
                        plt.text(1, 1, "Dependent On: " + '\n'.join([str(i) for i in dic['dependencies']]), horizontalalignment='right',verticalalignment='top', transform=plt.gca().transAxes)

                if 'rotate' in dic:
                    plt.xticks(rotation=dic['rotate'])
            plt.legend(loc='upper left')
            #fig.tight_layout()
            GraphData.saveFigure(fig,40,20, os.path.join(storeLocation,name))

        @staticmethod
        def comparisonSeriesPlotLog(dataDicList, storeLocation):
            """
            Draw one graph comparing the data in the list of Series

            Parameters
            ----------
            dataDicList : list of pandas series
                contains data to be graphed
            storeLocation: string
                location where graphs should be stored
            Returns
            -------
            """
            if 'title' in dataDicList[0]:
                name = dataDicList[0]['title']
            else:
                name = dataDicList[0]['data'].name

            fig = GraphData.createFigure(name)
            for dic in dataDicList:
                if 'style' in dic:
                        plt.plot(dic['data'], dic['style'], label=dic['name'])
                else:
                        plt.plot(dic['data'], label = dic['name'])
                if 'dependencies' in dic:
                        plt.text(1, 1, "Dependent On: " + '\n'.join([str(i) for i in dic['dependencies']]), horizontalalignment='right',verticalalignment='top', transform=plt.gca().transAxes)

                if 'rotate' in dic:
                    plt.xticks(rotation=dic['rotate'])
            plt.legend(loc='upper left')
            plt.yscale('log')
            GraphData.saveFigure(fig,40,20, os.path.join(storeLocation,name))


        @staticmethod
        def formPlotDictionary(name, data):
            return {'name': name, 'data': data}
