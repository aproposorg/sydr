
import pickle
from click import prompt
import numpy as np
import pandas as pd
import configparser
from bokeh.io import save, output_file
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import Div, ColumnDataSource, HoverTool, BoxSelectTool, LassoSelectTool
from bokeh.models.widgets import DataTable, TableColumn, Tabs, Panel

from gnsstools.ephemeris import Ephemeris

from .rfsignal import RFSignal
from .satellite import Satellite
from .gnsssignal import SignalType
from .channel.abstract import ChannelState

class Visualisation:

    def __init__(self, configfile, rfSignal:RFSignal, gnssSignals:dict):
        self.gnssSignals = gnssSignals
        self.rfSignal = rfSignal

        config = configparser.ConfigParser()
        config.read(configfile)

        self.outfolder = config.get('DEFAULT', 'outfolder')

        # Bokeh parameters
        self.backgroundColor = "#fafafa"
        self.tooltips = [("x", "$x{8.3f}"), ("y", "$y{8.3f}")]
        pass

    # -------------------------------------------------------------------------

    def run(self, signalType:SignalType):

        for prn, sat in self.satelliteDict.items():
            self.plotSatelliteDSP(sat, signalType)

        return

    # -------------------------------------------------------------------------

    def plotSatelliteDSP(self, satellite:Satellite, signalType: SignalType):

        # Acquisition
        acqLayout = self.getAcquisitionLayout(satellite, signalType)
        acqTab = Panel(child=acqLayout, title="Acquisition")

        # Tracking
        trackLayout = self.getTrackingLayout(satellite, signalType)
        trackTab = Panel(child=trackLayout, title="Tracking")

        # Decoding
        decodingLayout = self.getDecodingLayout(satellite, signalType)
        decodingTab = Panel(child=decodingLayout, title="Decoding")
        
        # Make tabs
        tabs = Tabs(tabs=[acqTab, trackTab, decodingTab])

        # Save file
        output_file(f'{self.outfolder}dsp_analysis_G{satellite.satelliteID}.html', title=f'DSP analysis G{satellite.satelliteID}')
        save(tabs)

        return

    # -------------------------------------------------------------------------

    def getAcquisitionLayout(self, satellite:Satellite, signalType: SignalType):
        """
        TODO
        """

        gnssSignal = self.gnssSignals[signalType]
        epochs = satellite.dspEpochs[signalType]

        html = f"<h2>Acquisition results summary</h2>\n<h3>{signalType}</h3>"
        tabTitle = Div(text=html)

        # Find first successful acquisition
        idx = epochs.state.index(ChannelState.ACQUIRING)
        dsp = epochs.dspMeasurements[idx]

        dopplerRange = gnssSignal.config.getfloat('ACQUISITION', 'doppler_range')
        dopplerSteps = gnssSignal.config.getfloat('ACQUISITION', 'doppler_steps')
        frequencyBins = np.arange(-dopplerRange, dopplerRange, dopplerSteps)
        codeBins =  np.linspace(0, gnssSignal.codeBits, np.size(dsp.correlationMap, axis=1))

        # Frequency correlation
        figAcqDoppler = figure(title="Frequency correlation", background_fill_color=self.backgroundColor)
        figAcqDoppler.line(x=frequencyBins, y=dsp.correlationMap[:, dsp.idxCode])
        figAcqDoppler.yaxis.axis_label = "Correlation amplitude"
        figAcqDoppler.xaxis.axis_label = "Frequency range"
        figAcqDoppler.add_tools(HoverTool(tooltips=self.tooltips))

        # Code correlation
        figAcqCode = figure(title="Code correlation", background_fill_color=self.backgroundColor)
        figAcqCode.line(x=codeBins, y=dsp.correlationMap[dsp.idxDoppler, :])
        figAcqCode.yaxis.axis_label = "Correlation amplitude"
        figAcqCode.xaxis.axis_label = "Code range"
        figAcqCode.add_tools(HoverTool(tooltips=self.tooltips))

        # 3D Correlation map
        # TODO No 3D support in Bokeh

        # Results table
        titleResults = Div(text="<h3>Detailed results<h3>")
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
        
        # Parameter table
        titleParameters = Div(text="<h3>Detailed configuration<h3>")
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
        
        acqLayout = layout([[tabTitle], \
                           [figAcqDoppler, figAcqCode],
                           [[titleResults, tableResults],[titleParameters, tableParameters]]])

        return acqLayout

    # -------------------------------------------------------------------------

    def getTrackingLayout(self, satellite:Satellite, signalType: SignalType):
        """
        TODO
        """

        tools = [HoverTool(tooltips=self.tooltips), 'box_select', 'lasso_select', \
            'pan', 'wheel_zoom', 'box_zoom,reset']

        gnssSignal = self.gnssSignals[signalType]
        epochs = satellite.dspEpochs[signalType]

        html = f"<h2>Tracking results summary</h2>\n<h3>{signalType}</h3>"
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

        # create a column data source for the plots to share
        # This is to share the lasso selection
        source = ColumnDataSource(data=dict(time=time, iprompt=iprompt, qprompt=qprompt, dll=dll, pll=pll))
        
        # I/Q plots
        height=400
        width=400
        figConstellation = figure(
            title="Constellation diagram", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools)
        figConstellation.scatter(x='iprompt', y='qprompt', source=source, size=10, marker='dot')
        figConstellation.yaxis.axis_label = "Quadraphase (Q)"
        figConstellation.xaxis.axis_label = "In-Phase (I)"

        # Parameter table
        tableParameters = self.getParameterTable(gnssSignal.config, 'TRACKING')
        
        height=300
        width=500
        # DLL
        figDLL = figure(
            title="DLL", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools)
        figDLL.line(x='time', y='dll', source=source)
        figDLL.yaxis.axis_label = "Code frequency jitter [Hz]"
        figDLL.xaxis.axis_label = "Time [s]"

        # PLL
        figPLL = figure(
            title="PLL", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools,
            x_range=figDLL.x_range)
        figPLL.line(x='time', y='pll', source=source)
        figPLL.yaxis.axis_label = "Doppler frequency jitter [Hz]"
        figPLL.xaxis.axis_label = "Time [s]"

        height=300
        width=1000
        # In-Phase (I) Prompt
        figIPrompt = figure(
            title="In-Phase (I) Prompt", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools,
            x_range=figDLL.x_range)
        figIPrompt.line(x='time', y='iprompt', source=source)
        figIPrompt.scatter(x='time', y='iprompt', source=source, marker='dot')
        figIPrompt.yaxis.axis_label = "Correlation amplitude"
        figIPrompt.xaxis.axis_label = "Time [s]"

        # Quadraphase (Q) Prompt
        figQPrompt = figure(
            title="Quadraphase (Q) Prompt",\
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools,
            x_range=figDLL.x_range)
        figQPrompt.line(x='time', y='qprompt', source=source)
        figQPrompt.scatter(x='time', y='qprompt', source=source, marker='dot')
        figQPrompt.yaxis.axis_label = "Correlation amplitude"
        figQPrompt.xaxis.axis_label = "Time [s]"

        trackLayout = layout([[tabTitle],
                              [figConstellation, tableParameters],\
                              [figDLL, figPLL], \
                              [figIPrompt], [figQPrompt]])
        
        return trackLayout

    # -------------------------------------------------------------------------

    def getDecodingLayout(self, satellite:Satellite, signalType: SignalType):

        tools = [HoverTool(tooltips=self.tooltips), 'box_select', 'lasso_select', \
            'pan', 'wheel_zoom', 'box_zoom,reset']

        gnssSignal = self.gnssSignals[signalType]
        navMessage = satellite.navMessage[signalType]

        html = f"<h2>Decoding results summary</h2>\n<h3>{signalType}</h3>"
        tabTitle = Div(text=html)

        time = np.array(navMessage.time) / 1e3

        height=300
        width=1000
        # Decoded bits
        figBits = figure(
            title="Navigation message bits", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools)
        figBits.line(x=time, y=navMessage.bits)
        figBits.yaxis.axis_label = "Bits"
        figBits.xaxis.axis_label = "Time [s]"

        # Content table
        # TODO
        
        decodingLayout = layout([[tabTitle],
                              [figBits]])
        
        return decodingLayout

    # -------------------------------------------------------------------------

    def getParameterTable(self, config, section):
        """
        TODO
        """
        parameters = [key for key, value in config.items(section)]
        values = [value for key, value in config.items(section)]
        dfParameters = pd.DataFrame({
            'Parameters' : parameters,
            'Values' : values
        })
        source = ColumnDataSource(dfParameters)
        columns = [
            TableColumn(field="Parameters", title="Parameters"),
            TableColumn(field="Values", title="Values")]
        table = DataTable(source=source, columns=columns, autosize_mode="fit_columns")

        return table

    # -------------------------------------------------------------------------

    def importSatellites(self, picklefile):
        with open(picklefile, 'rb') as f:
                self.satelliteDict = pickle.load(f)
        return 




