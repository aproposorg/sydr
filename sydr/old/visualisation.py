
import pickle
import numpy as np
import pandas as pd
import configparser
from bokeh.io import save, output_file
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import Div, ColumnDataSource, HoverTool, PrintfTickFormatter, Range1d
from bokeh.models.widgets import DataTable, TableColumn, Tabs, Panel
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pymap3d as pm

from sydr.utils.constants import SPEED_OF_LIGHT
from sydr.receiver.receiver_abstract import ReceiverAbstract
from sydr.signal.rfsignal import RFSignal
from sydr.space.satellite import Satellite
from sydr.utils.enumerations import GNSSSignalType
from sydr.channel.channel_abstract import ChannelState

class Visualisation:

    receiver : ReceiverAbstract

    def __init__(self, configfile, rfSignal:RFSignal, gnssSignals:dict):
        self.gnssSignals = gnssSignals
        self.rfSignal = rfSignal

        config = configparser.ConfigParser()
        config.read(configfile)

        self.outfolder = config.get('DEFAULT', 'outfolder')

        # Receiver
        self.referencePosition = np.array([
            config.getfloat('RECEIVER', 'reference_position_x'), \
            config.getfloat('RECEIVER', 'reference_position_y'), \
            config.getfloat('RECEIVER', 'reference_position_z')])

        # Bokeh parameters
        self.backgroundColor = "#fafafa"
        self.tooltips = [("x", "$x{8.3f}"), ("y", "$y{8.3f}")]
        pass

    # -------------------------------------------------------------------------

    def setDatabase(self, database):
        self.database = database
        return

    # -------------------------------------------------------------------------

    def run(self, signalType:GNSSSignalType):

        for prn, sat in self.satelliteDict.items():
            self.plotSatelliteDSP(sat, signalType)
        
        # Position analysis
        self.plotReceiver()

        return

    # -------------------------------------------------------------------------

    def plotReceiver(self):

        specs = [[{'type': 'mapbox',"rowspan": 2}, {'type': 'xy',"rowspan": 2}],
                 [None, None],
                 [{'type': 'xy',"colspan": 2}, None],
                 [{'type': 'xy',"colspan": 2}, None],
                 [{'type': 'xy',"colspan": 2}, None],
                 [{'type': 'xy',"colspan": 2}, None]]

        titles = ["Map", "North/East",
                  "North [m]",
                  "East [m]",
                  "Up [m]",
                  "Recevier clock bias [s]"]
        
        # Retrieve from database
        positionList = self.receiver.database.fetchPositions()

        recpos = []
        recclk = []
        time = []
        for position in positionList:
            time.append(position.time.datetime)
            recpos.append(position.coordinate.vecpos())
            recclk.append(position.clockError)

        #recpos = np.array(recpos)
        recclk = np.array(recclk)
        
        # recpos = np.array(self.receiver.receiverPosition)
        # recclk = np.array(self.receiver.receiverClockError)
        refpos = pm.ecef2geodetic( \
            self.referencePosition[0], \
            self.referencePosition[1], \
            self.referencePosition[2], deg=True)
        
        # Convert to ENU
        enu = []
        llh = []
        for coord in recpos:
            enu.append(pm.ecef2enu(coord[0], coord[1], coord[2], refpos[0], refpos[1], refpos[2]))
            llh.append(pm.ecef2geodetic(coord[0], coord[1], coord[2], ell=None, deg=True))
        enu = np.array(enu)
        llh = np.array(llh)

        #time = self.receiver.measurementTimeList

        fig = make_subplots(6, 2,\
                start_cell="top-left",
                specs=specs, 
                subplot_titles=titles,
                vertical_spacing=0.1, 
                horizontal_spacing = 0.1)

        # Map box
        figpos = (1,1)
        fig.add_trace(go.Scattermapbox(lat=llh[:,0], lon=llh[:,1], mode='markers'), \
            row=figpos[0], col=figpos[1])
        
        
        fig.update_layout(
            margin ={'l':0,'t':0,'b':0,'r':0},
            mapbox = {
                'center': {'lon': refpos[1], 'lat': refpos[0]},
                'style': "open-street-map",
                'zoom': 15})

        # North/East view
        figpos = (1,2)
        fig.add_trace(go.Scatter(x=enu[:,0], y=enu[:,1], mode="markers",
                                line=dict(width=2)), row=figpos[0], col=figpos[1])
        fig.update_xaxes(title_text="East [m]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(title_text="North [m]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray', scaleanchor="x", scaleratio=1)

        # East
        figpos = (3,1)
        fig.add_trace(go.Scatter(x=time, y=enu[:,0], mode="lines+markers",
                                line=dict(width=2)), row=figpos[0], col=figpos[1])
        fig.update_xaxes(title_text="Time [s]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(title_text="East [m]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # North
        figpos = (4,1)
        fig.add_trace(go.Scatter(x=time, y=enu[:,1], mode="lines+markers",
                                line=dict(width=2)), row=figpos[0], col=figpos[1])
        fig.update_xaxes(title_text="Time [s]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(title_text="North [m]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # Up
        figpos = (5,1)
        fig.add_trace(go.Scatter(x=time, y=enu[:,2], mode="lines+markers",
                                line=dict(width=2)), row=figpos[0], col=figpos[1])
        fig.update_xaxes(title_text="Time [s]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(title_text="Up [m]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')

        # Receiver clock bias
        figpos = (6,1)
        fig.add_trace(go.Scatter(x=time, y=recclk/SPEED_OF_LIGHT, mode="lines+markers",
                                line=dict(width=2)), row=figpos[0], col=figpos[1])
        fig.update_xaxes(title_text="Time [s]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(title_text="Bias [s]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')

        fig.update_layout(title=f"Navigation solution", margin=dict(l=50,r=50,b=100,t=100,pad=4),showlegend=False) 
        fig.write_html(f"./{self.outfolder}/navigation.html")

        return
    
    # -------------------------------------------------------------------------

    # def getMapLayout(self):

    #     html = f"<h2>Positioning results summary</h2>"
    #     tabTitle = Div(text=html)
        

    #     return

    # -------------------------------------------------------------------------

    def plotSatelliteDSP(self, satellite:Satellite, signalType:GNSSSignalType):

        # Acquisition
        acqLayout = self.getAcquisitionLayout(satellite, signalType)
        acqTab = Panel(child=acqLayout, title="Acquisition")

        # Tracking
        trackLayout = self.getTrackingLayout(satellite, signalType)
        trackTab = Panel(child=trackLayout, title="Tracking")

        # # Decoding
        # decodingLayout = self.getDecodingLayout(satellite, signalType)
        # decodingTab = Panel(child=decodingLayout, title="Decoding")
        
        # Make tabs
        tabs = Tabs(tabs=[acqTab, trackTab])

        # Save file
        output_file(f'{self.outfolder}dsp_analysis_G{satellite.svid}.html', title=f'DSP analysis G{satellite.svid}')
        save(tabs)

        return

    # -------------------------------------------------------------------------

    def getAcquisitionLayout(self, satellite:Satellite, signalType: GNSSSignalType):
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

        titleFontSize = '16pt'
        tickFontSize = '16pt'
        axisFontSize = '16pt'
        lineWidth = 2

        dopplerRange = gnssSignal.config.getfloat('ACQUISITION', 'doppler_range')
        dopplerSteps = gnssSignal.config.getfloat('ACQUISITION', 'doppler_steps')
        frequencyBins = np.arange(-dopplerRange, dopplerRange, dopplerSteps)
        codeBins =  np.linspace(0, gnssSignal.codeBits, np.size(dsp.correlationMap, axis=1))

        height = 300
        width = 500
        # Frequency correlation
        figAcqDoppler = figure(title="Frequency correlation", background_fill_color=self.backgroundColor, \
            height=height, width=width)
        figAcqDoppler.line(x=frequencyBins, y=dsp.correlationMap[:, dsp.idxCode], line_width=lineWidth)
        figAcqDoppler.yaxis.axis_label = "Correlation amplitude"
        figAcqDoppler.xaxis.axis_label = "Frequency range [Hz]"
        figAcqDoppler.add_tools(HoverTool(tooltips=self.tooltips))
        figAcqDoppler.title.text_font_size = titleFontSize
        figAcqDoppler.xaxis.major_label_text_font_size = tickFontSize
        figAcqDoppler.xaxis.axis_label_text_font_size = axisFontSize
        figAcqDoppler.yaxis.major_label_text_font_size = tickFontSize
        figAcqDoppler.yaxis.axis_label_text_font_size = axisFontSize
        figAcqDoppler.yaxis.formatter=PrintfTickFormatter(format="%0e")

        # Code correlation
        figAcqCode = figure(title="Code correlation", background_fill_color=self.backgroundColor, \
            height=height, width=width)
        figAcqCode.line(x=codeBins, y=dsp.correlationMap[dsp.idxDoppler, :], line_width=lineWidth)
        figAcqCode.yaxis.axis_label = "Correlation amplitude"
        figAcqCode.xaxis.axis_label = "Code range [chip]"
        figAcqCode.add_tools(HoverTool(tooltips=self.tooltips))
        figAcqCode.title.text_font_size = titleFontSize
        figAcqCode.xaxis.major_label_text_font_size = tickFontSize
        figAcqCode.xaxis.axis_label_text_font_size = axisFontSize
        figAcqCode.yaxis.major_label_text_font_size = tickFontSize
        figAcqCode.yaxis.axis_label_text_font_size = axisFontSize
        figAcqCode.yaxis.formatter=PrintfTickFormatter(format="%0e")

        # 3D Correlation map
        # TODO No 3D support in Bokeh

        # Results table
        titleResults = Div(text="<h3>Detailed results<h3>")
        samplesPerCode = round(self.rfSignal.samplingFrequency / (gnssSignal.codeFrequency / gnssSignal.codeBits))
        dfResults = pd.DataFrame({
            'Parameters' : ['PRN', 'Signal', 'Metric', 'Doppler [Hz]', 'Code sample', 'Code phase (bits)'],
            'Values' : [f'G{satellite.svid}', f'{gnssSignal.signalType}', f'{dsp.acquisitionMetric:>6.2}',\
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

    def getTrackingLayout(self, satellite:Satellite, signalType: GNSSSignalType):
        """
        TODO
        """

        titleFontSize = '16pt'
        tickFontSize = '16pt'
        axisFontSize = '16pt'
        lineWidth = 2

        tools = [HoverTool(tooltips=self.tooltips), 'box_select', 'lasso_select', \
            'pan', 'wheel_zoom', 'box_zoom,reset', 'save']

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
        figDLL.line(x='time', y='dll', source=source, line_width=lineWidth)
        figDLL.yaxis.axis_label = "Code frequency jitter [Hz]"
        figDLL.xaxis.axis_label = "Time [s]"
        figDLL.title.text_font_size = titleFontSize
        figDLL.xaxis.major_label_text_font_size = tickFontSize
        figDLL.xaxis.axis_label_text_font_size = axisFontSize
        figDLL.yaxis.major_label_text_font_size = tickFontSize
        figDLL.yaxis.axis_label_text_font_size = axisFontSize
        #figDLL.yaxis.formatter=PrintfTickFormatter(format="%0e")

        # PLL
        figPLL = figure(
            title="PLL", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools,
            x_range=figDLL.x_range)
        figPLL.line(x='time', y='pll', source=source, line_width=lineWidth)
        figPLL.yaxis.axis_label = "Doppler frequency jitter [Hz]"
        figPLL.xaxis.axis_label = "Time [s]"
        figPLL.title.text_font_size = titleFontSize
        figPLL.xaxis.major_label_text_font_size = tickFontSize
        figPLL.xaxis.axis_label_text_font_size = axisFontSize
        figPLL.yaxis.major_label_text_font_size = tickFontSize
        figPLL.yaxis.axis_label_text_font_size = axisFontSize
        #figPLL.yaxis.formatter=PrintfTickFormatter(format="%0e")

        height=300
        width=1000
        # In-Phase (I) Prompt
        figIPrompt = figure(
            title="In-Phase (I) Prompt", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools,
            x_range=figDLL.x_range,
            y_range=Range1d(-20e3, 20e3))
        figIPrompt.line(x='time', y='iprompt', source=source, line_width=lineWidth)
        figIPrompt.scatter(x='time', y='iprompt', source=source, marker='dot')
        figIPrompt.yaxis.axis_label = "Correlation amplitude"
        figIPrompt.xaxis.axis_label = "Time [s]"
        figIPrompt.title.text_font_size = titleFontSize
        figIPrompt.xaxis.major_label_text_font_size = tickFontSize
        figIPrompt.xaxis.axis_label_text_font_size = axisFontSize
        figIPrompt.yaxis.major_label_text_font_size = tickFontSize
        figIPrompt.yaxis.axis_label_text_font_size = axisFontSize

        # Quadraphase (Q) Prompt
        figQPrompt = figure(
            title="Quadraphase (Q) Prompt",\
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools,
            x_range=figDLL.x_range,
            y_range=Range1d(-20e3, 20e3))
        figQPrompt.line(x='time', y='qprompt', source=source, line_width=lineWidth)
        figQPrompt.scatter(x='time', y='qprompt', source=source, marker='dot')
        figQPrompt.yaxis.axis_label = "Correlation amplitude"
        figQPrompt.xaxis.axis_label = "Time [s]"
        figQPrompt.title.text_font_size = titleFontSize
        figQPrompt.xaxis.major_label_text_font_size = tickFontSize
        figQPrompt.xaxis.axis_label_text_font_size = axisFontSize
        figQPrompt.yaxis.major_label_text_font_size = tickFontSize
        figQPrompt.yaxis.axis_label_text_font_size = axisFontSize

        trackLayout = layout([[tabTitle],
                              [figConstellation, tableParameters],\
                              [figDLL, figPLL], \
                              [figIPrompt], [figQPrompt]])
        
        return trackLayout

    # -------------------------------------------------------------------------

    def getDecodingLayout(self, satellite:Satellite, signalType: GNSSSignalType):

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

    def plotMeasurements(self):

        measurementList = self.receiver.database.fetchPositions()

        return

    # -------------------------------------------------------------------------

    def importSatellites(self, picklefile):
        with open(picklefile, 'rb') as f:
                self.satelliteDict = pickle.load(f)
        return 

    # -------------------------------------------------------------------------

    def importReceiver(self, picklefile):
        with open(picklefile, 'rb') as f:
                self.receiver = pickle.load(f)
        return 




