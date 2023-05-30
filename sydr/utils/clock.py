

import copy
from datetime import timedelta

from sydr.utils.time import Time

class Clock:

    absoluteTime:Time
    relativeTime:float

    isInitialised : bool

    def __init__(self):
        self.absoluteTime = Time()
        self.relativeTime = 0.0
        self.isInitialised = False
        return

    def __str__(self):
        return str(self.absoluteTime)

    def setAbsoluteTime(self, time:Time):
        self.absoluteTime = time
        self.isInitialised = True
        return

    def addTime(self, seconds):
        self.absoluteTime.setDatetime(self.absoluteTime.datetime + timedelta(seconds=seconds))
        self.relativeTime += seconds
        return