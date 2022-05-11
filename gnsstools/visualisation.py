
from operator import index
import pickle
import numpy as np
import pandas as pd
from bokeh.io import save, output_file
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import Div, ColumnDataSource, HoverTool
from bokeh.models.widgets import DataTable, TableColumn, Tabs, Panel

from gnsstools.measurements import DSPmeasurement
from gnsstools.rfsignal import RFSignal

from .satellite import Satellite
from .gnsssignal import SignalType
from .channel.abstract import ChannelState

class Visualisation:

    def __init__(self, rfSignal:RFSignal, gnssSignals:dict):
        self.gnssSignals = gnssSignals
        self.rfSignal = rfSignal
        pass

    def run(self, signalType:SignalType):

        for prn, sat in self.satelliteDict.items():
            self.plotSatelliteDSP(sat, signalType)

        return

    def plotSatelliteDSP(self, satellite:Satellite, signalType: SignalType):
        gnssSignal = self.gnssSignals[signalType]

        # Defining the figures
        backgroundColor = "#fafafa"
        tooltips = [("x", "$x{8.3f}"), ("y", "$y{8.3f}")]
        tabsList = []

        # Acquisition
        epochs = satellite.dspEpochs[signalType]

        # Find first successful acquisition
        idx = epochs.state.index(ChannelState.ACQUIRED)
        dsp = epochs.dspMeasurements[idx]

        dopplerRange = gnssSignal.config.getfloat('ACQUISITION', 'doppler_range')
        dopplerSteps = gnssSignal.config.getfloat('ACQUISITION', 'doppler_steps')
        frequencyBins = np.arange(-dopplerRange, dopplerRange, dopplerSteps)
        codeBins =  np.linspace(0, gnssSignal.codeBits, np.size(dsp.correlationMap, axis=1))

        # Frequency correlation
        figAcqDoppler = figure(title="Frequency correlation", background_fill_color=backgroundColor)
        figAcqDoppler.line(x=frequencyBins, y=dsp.correlationMap[:, dsp.idxCode])
        figAcqDoppler.yaxis.axis_label = "Correlation amplitude"
        figAcqDoppler.xaxis.axis_label = "Frequency range"
        figAcqDoppler.add_tools(HoverTool(tooltips=tooltips))

        # Code correlation
        figAcqCode = figure(title="Code correlation", background_fill_color=backgroundColor)
        figAcqCode.line(x=codeBins, y=dsp.correlationMap[dsp.idxDoppler, :])
        figAcqCode.yaxis.axis_label = "Correlation amplitude"
        figAcqCode.xaxis.axis_label = "Code range"
        figAcqCode.add_tools(HoverTool(tooltips=tooltips))

        # 3D Correlation map
        # TODO No 3D support in Bokeh

        # Results table
        samplesPerCode = round(self.rfSignal.samplingFrequency / (gnssSignal.codeFrequency / gnssSignal.codeBits))
        dfResults = pd.DataFrame({
            'Parameters' : ['PRN', 'Signal', 'Metric', 'Doppler [Hz]', 'Code sample', 'Code phase (bits)'],
            'Values' : [f'G{satellite.satelliteID}', f'{gnssSignal.signalType}', f'{dsp.acquisitionMetric:>6.2}',\
                f'{dsp.dopplerFrequency}', f'{dsp.codeShift}', \
                f'{dsp.codeShift * gnssSignal.codeBits / samplesPerCode:>8.2f}']
        })
        source = ColumnDataSource(dfResults)
        columns = [
            TableColumn(field="Parameters", title="Parameters"),
            TableColumn(field="Values", title="Values")]
        tableResults = DataTable(source=source, columns=columns)
        titleResults = Div(text="<h3>Detailed configuration<h3>")

        # Parameter table
        parameters = [key for key, value in gnssSignal.config.items('ACQUISITION')]
        values = [value for key, value in gnssSignal.config.items('ACQUISITION')]
        dfParameters = pd.DataFrame({
            'Parameters' : parameters,
            'Values' : values
        })
        source = ColumnDataSource(dfParameters)
        columns = [
            TableColumn(field="Parameters", title="Parameters"),
            TableColumn(field="Values", title="Values")]
        tableParameters = DataTable(source=source, columns=columns)
        titleParameters = Div(text="<h3>Detailed results<h3>")

        # 
        html = """<h3>Acquisition results</h3>"""
        tabTitle = Div(text=html)

        acqLayout = layout([[tabTitle], \
                           [figAcqDoppler, figAcqCode],
                           [[titleResults, tableResults],[titleParameters, tableParameters]]
                           ])

        acqTab = Panel(child=acqLayout, title="Acquisition")

        # Tracking
        html = """<h3>Tracking results</h3>"""
        tabTitle = Div(text=html)
        
        time = np.array(epochs.time) / 1e3
        size = time.size
        iprompt = np.full(size, np.nan)
        qprompt = np.full(size, np.nan)
        dll = np.full(size, np.nan)
        pll = np.full(size, np.nan)
        i = 0
        for dsp in epochs.dspMeasurements:
            if dsp.state == ChannelState.TRACKING:
                iprompt[i] = dsp.iPrompt
                qprompt[i] = dsp.qPrompt
                dll[i] = dsp.dll
                pll[i] = dsp.pll
                i += 1
        
        # I/Q plots
        
        height=300
        width=500
        # DLL
        figTrackDLL = figure(
            title="DLL", \
            background_fill_color=backgroundColor,\
            height=height, width=width)
        figTrackDLL.line(x=time, y=dll)
        figTrackDLL.yaxis.axis_label = "Doppler frequency jitter [Hz]"
        figTrackDLL.xaxis.axis_label = "Time [s]"
        figTrackDLL.add_tools(HoverTool(tooltips=tooltips))

        # PLL
        figTrackPLL = figure(
            title="PLL", \
            background_fill_color=backgroundColor,\
            height=height, width=width,
            x_range=figTrackDLL.x_range)
        figTrackPLL.line(x=time, y=pll)
        figTrackPLL.yaxis.axis_label = "Doppler frequency jitter [Hz]"
        figTrackPLL.xaxis.axis_label = "Time [s]"
        figTrackPLL.add_tools(HoverTool(tooltips=tooltips))

        height=300
        width=1000
        # In-Phase (I) Prompt
        figTrackIPrompt = figure(
            title="In-Phase (I) Prompt", \
            background_fill_color=backgroundColor,\
            height=height, width=width,
            x_range=figTrackDLL.x_range)
        figTrackIPrompt.line(x=time, y=iprompt)
        figTrackIPrompt.yaxis.axis_label = "Correlation amplitude"
        figTrackIPrompt.xaxis.axis_label = "Time [s]"
        figTrackIPrompt.add_tools(HoverTool(tooltips=tooltips))

        # Quadraphase (Q) Prompt
        figTrackQPrompt = figure(
            title="Quadraphase (Q) Prompt",\
            background_fill_color=backgroundColor,\
            height=height, width=width,
            x_range=figTrackDLL.x_range)
        figTrackQPrompt.line(x=time, y=qprompt)
        figTrackQPrompt.yaxis.axis_label = "Correlation amplitude"
        figTrackQPrompt.xaxis.axis_label = "Time [s]"
        figTrackQPrompt.add_tools(HoverTool(tooltips=tooltips))

        trackLayout = layout([[tabTitle],[figTrackDLL, figTrackPLL], [figTrackIPrompt], [figTrackQPrompt]])
        trackTab = Panel(child=trackLayout, title="Tracking")

        # Make tabs
        tabs = Tabs(tabs=[acqTab, trackTab])
        
        output_file('dsp_analysis.html', title='DSP analysis')
        save(tabs)

        pass

    def importSatellites(self, picklefile):
        with open(picklefile, 'rb') as f:
                self.satelliteDict = pickle.load(f)
        return 

    def makeHTMLTable(df:pd.DataFrame):

        return html





