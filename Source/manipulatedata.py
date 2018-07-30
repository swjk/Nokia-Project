import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config as cfg

class DataManipulation(object):
    @staticmethod
    def groupDataByDay(data):
        return data.groupby(data.index.dayofyear, sort=False)
    @staticmethod
    def groupDataByDayName(data):
        return data.groupby(data.index.weekday, sort=False)
    @staticmethod
    def groupDataByTime(data):
        return data.groupby(data.index.time, sort=False)
    @staticmethod
    def groupDataByDate(data):
        return data.groupby(data.index.date, sort=False)
