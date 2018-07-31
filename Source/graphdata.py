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
            plt.savefig(location, dpi=100, format='png')

        @staticmethod
        def createFigure(title):
            """
            Create Figure with specified title
            """
            fig = plt.figure()
            plt.title(title)
            return fig

        @staticmethod
        def comparisonPlot(dataDicList, storeLocation):
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
        def formPlotDictionary(name, data):
            return {'name': name, 'data': data}
