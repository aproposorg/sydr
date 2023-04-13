
from datetime import datetime
import logging
import numpy as np
import configparser
import panel as pn
import pandas as pd
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.models import Div, ColumnDataSource, HoverTool, PrintfTickFormatter, Range1d
from bokeh.models.widgets import DataTable, TableColumn
import holoviews as hv
import pymap3d as pm
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from core.utils.enumerations import GNSSSignalType
from core.io.database import DatabaseHandler
from core.receiver.receiver import Receiver
from core.signal.rfsignal import RFSignal
from core.utils.constants import GPS_L1CA_CODE_SIZE_BITS

hv.extension("bokeh")
pn.extension(sizing_mode="stretch_width")

class Visualisation:

    receiver : Receiver
    database : DatabaseHandler

    def __init__(self, configuration:configparser.ConfigParser):
        """
        """

        # Configuration
        self.outfolder = configuration['DEFAULT']['outfolder']
        self.referencePosition = np.array([
            float(configuration['DEFAULT']['reference_position_x']),
            float(configuration['DEFAULT']['reference_position_y']),
            float(configuration['DEFAULT']['reference_position_z'])])
        
        # Channel configuration
        # TODO Modify for more channels
        self.channelConfig = configparser.ConfigParser()
        self.channelConfig.read(configuration['CHANNELS']['gps_l1ca'])

        # RF Signal
        self.rfSignal = RFSignal(configuration['RFSIGNAL'])

        # Database
        receiverName = str(configuration['DEFAULT']['name'])
        self.database = DatabaseHandler(f"{self.outfolder}/{receiverName}.db", overwrite=False)

        # Bokeh parameters
        self.backgroundColor = "#fafafa"
        self.tooltips = [("x", "$x{8.3f}"), ("y", "$y{8.3f}")]

        logging.getLogger(__name__).info(f"VisualisationV2 initialized.")

        return

    # -------------------------------------------------------------------------

    def run(self):
        mainTabs = pn.Tabs()

        # # Measurement Tab   
        measurementTab = self._getMeasurementsTab()
        mainTabs.append(('Measurements', measurementTab))

        # Navigation Tab
        navigationTab = self._getNavigationTab()
        mainTabs.append(('Navigation', navigationTab))

        _filepath = f"./{self.outfolder}/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        mainTabs.save(_filepath, embed=True)

        logging.getLogger(__name__).info(f"HTML report created and saved at [{_filepath}].")

        # Processing time 
        # self._processingTime()

        return
    
    # -------------------------------------------------------------------------

    def _getMeasurementsTab(self):

        # Fetch satellite list
        channelList = self.database.fetchTable('channel')
        satelliteList = {f"{channel['satellite_id']}":channel["id"] for channel in channelList}
        checkboxes_prn = pn.widgets.ToggleGroup(options=satelliteList, behavior='radio', button_type="success", width=200)

        selections = pn.Column('### Satellites and signals', checkboxes_prn)
        #layout = pn.Row(selections, tabs)

        # Function definition for result handling
        @pn.depends(checkboxes_prn.param.value)
        def tabs(channelID):
            
            acqLayout   = self._getAcquisitionLayout(channelID)
            trackLayout = self._getTrackingLayout(channelID)
            measLayout  = self._getGNSSMeasurementLayout(channelID)
            
            return pn.Tabs(
                ('Acquisition', acqLayout),
                ('Tracking', trackLayout),
                ('GNSS Measurements', measLayout)
            )
        layout = pn.Column(selections, tabs)

        return layout
    
    # -------------------------------------------------------------------------

    def _getAcquisitionLayout(self, channelID):
        """
        TODO
        """

        # Parameters 
        # TODO Move to config file
        titleFontSize = '16pt'
        tickFontSize = '16pt'
        axisFontSize = '16pt'
        lineWidth = 2

        html = f"<h2>Acquisition results summary</h2>\n<h3>{GNSSSignalType.GPS_L1_CA}</h3>"
        tabTitle = Div(text=html)

        # Retrieve data and prepare for plotting
        dataList = self.database.fetchAcquisition(channelID)

        # Only take the first acquisition results
        # TODO Find a way to display the multiple results in case of re-acquisition
        acquisition = dataList[0]

        dopplerRange = float(self.channelConfig['ACQUISITION']['doppler_range'])
        dopplerSteps = float(self.channelConfig['ACQUISITION']['doppler_steps'])
        frequencyBins = np.arange(-dopplerRange, dopplerRange, dopplerSteps)
        codeBins =  np.linspace(0, GPS_L1CA_CODE_SIZE_BITS, np.size(acquisition["correlation_map"], axis=1))

        height = 300
        width = 500
        # Frequency correlation
        figAcqDoppler = figure(title="Frequency correlation", background_fill_color=self.backgroundColor, \
            height=height, width=width)
        figAcqDoppler.line(x=frequencyBins, y=acquisition["correlation_map"][:, acquisition["code_idx"]], line_width=lineWidth)
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
        figAcqCode.line(x=codeBins, y=acquisition["correlation_map"][acquisition["frequency_idx"], :], line_width=lineWidth)
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
        # titleResults = Div(text="<h3>Detailed results<h3>")
        # samplesPerCode = round(self.rfSignal.samplingFrequency / (self.gnssSignal.codeFrequency / self.gnssSignal.codeBits))
        # dfResults = pd.DataFrame({
        #     'Parameters' : ['PRN', 'Signal', 'Metric', 'Doppler [Hz]', 'Code sample', 'Code phase (bits)'],
        #     'Values' : [f'G{satellite.svid}', f'{self.gnssSignal.signalType}', f'{dsp.acquisitionMetric:>6.2}',\
        #         f'{dsp.dopplerFrequency}', f'{dsp.codeShift}', \
        #         f'{dsp.codeShift * self.gnssSignal.codeBits / samplesPerCode:>8.2f}']
        # })
        # source = ColumnDataSource(dfResults)
        # columns = [
        #     TableColumn(field="Parameters", title="Parameters"),
        #     TableColumn(field="Values", title="Values")]
        # tableResults = DataTable(source=source, columns=columns)
        
        # Parameter table
        titleParameters = Div(text="<h3>Detailed configuration<h3>")
        parameters = [key for key, value in self.channelConfig.items('ACQUISITION')]
        values = [value for key, value in self.channelConfig.items('ACQUISITION')]
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
                           [titleParameters, tableParameters]])

        return acqLayout

    # -------------------------------------------------------------------------

    def _getTrackingLayout(self, channelID):
        """
        TODO
        """

        # Parameters 
        # TODO Move to config file
        titleFontSize = '16pt'
        tickFontSize = '16pt'
        axisFontSize = '16pt'
        lineWidth = 2

        # Bokeh definitions
        tools = [HoverTool(tooltips=self.tooltips), 'box_select', 'lasso_select', \
            'pan', 'wheel_zoom', 'box_zoom,reset', 'save']
        
        html = f"<h2>Tracking results summary</h2>\n<h3>{GNSSSignalType.GPS_L1_CA}</h3>"
        tabTitle = Div(text=html)

        # Retrieve data
        dataList = self.database.fetchTracking(channelID)

        # Prepare data
        size = len(dataList)
        time = np.full(size, np.nan)
        iprompt = np.full(size, np.nan)
        qprompt = np.full(size, np.nan)
        dll = np.full(size, np.nan)
        pll = np.full(size, np.nan)
        i = 0
        for dsp in dataList:
            time[i]    = dsp["time_sample"] / self.rfSignal.samplingFrequency
            iprompt[i] = dsp["i_prompt"]
            qprompt[i] = dsp["q_prompt"]
            dll[i] = dsp["dll"]
            pll[i] = dsp["pll"]
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
        #tableParameters = self.getParameterTable(gnssSignal.config, 'TRACKING')
        
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
            y_range=Range1d(-25e3, 25e3))
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
            y_range=Range1d(-25e3, 25e3))
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
                              [figConstellation],\
                              [figDLL, figPLL], \
                              [figIPrompt], [figQPrompt]])
        
        return trackLayout

    # -------------------------------------------------------------------------

    def _getGNSSMeasurementLayout(self, channelID):

        # Parameters 
        # TODO Move to config file
        titleFontSize = '16pt'
        tickFontSize = '16pt'
        axisFontSize = '16pt'
        lineWidth = 2

        # Bokeh definitions
        tools = [HoverTool(tooltips=self.tooltips), 'box_select', 'lasso_select', \
            'pan', 'wheel_zoom', 'box_zoom,reset', 'save']

        html = f"<h2>Measurements results summary</h2>\n<h3>{GNSSSignalType.GPS_L1_CA}</h3>"
        tabTitle = Div(text=html)
        
        # Retrieve and prepare data
        # Pseudorange
        dataList = self.database.fetchMeasurements(channelID, "PSEUDORANGE")

        size = len(dataList)
        time = np.full(size, np.nan)
        pseudorange = np.full(size, np.nan)
        pseudorangeRate = np.full(size, np.nan)
        pseudorangeAcc = np.full(size, np.nan)
        doppler = np.full(size, np.nan)
        dopplerNoise = np.full(size, np.nan)
        idx = 0 
        for data in dataList:
            time[idx] = data["time_sample"] / self.rfSignal.samplingFrequency
            pseudorange[idx] = data["value"]
            idx += 1
        pseudorangeRate[1:] = pseudorange[1:] - pseudorange[:-1]
        pseudorangeAcc[2:]  = pseudorangeRate[2:] - pseudorangeRate[:-2]

        # Doppler
        dataList = self.database.fetchMeasurements(channelID, "DOPPLER")
        doppler = np.full(size, np.nan)
        dopplerNoise = np.full(size, np.nan)
        idx = 0 
        for data in dataList:
            time[idx] = data["time_sample"] / self.rfSignal.samplingFrequency
            doppler[idx] = data["value"]

            idx += 1
        dopplerNoise[1:] = doppler[1:] - doppler[:-1]

        source = ColumnDataSource(data=dict(time=time, pseudorange=pseudorange, pseudorangeRate=pseudorangeRate, \
            pseudorangeAcc=pseudorangeAcc, doppler=doppler, dopplerNoise=dopplerNoise))
        
        # Pseudorange
        height=300
        width=800
        figPseudorange = figure(
            title="Pseudorange", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools)
        figPseudorange.line(x='time', y='pseudorange', source=source, line_width=lineWidth)
        figPseudorange.yaxis.axis_label = "Range [m]"
        figPseudorange.xaxis.axis_label = "Time [s]"
        figPseudorange.title.text_font_size = titleFontSize
        figPseudorange.xaxis.major_label_text_font_size = tickFontSize
        figPseudorange.xaxis.axis_label_text_font_size = axisFontSize
        figPseudorange.yaxis.major_label_text_font_size = tickFontSize
        figPseudorange.yaxis.axis_label_text_font_size = axisFontSize

        # # Pseudorange rate
        # height=300
        # width=1000
        # figPseudoRate = figure(
        #     title="Pseudorange Rate", \
        #     background_fill_color=self.backgroundColor,\
        #     height=height, width=width, tools=tools)
        # figPseudoRate.line(x='time', y='pseudorangeRate', source=source, line_width=lineWidth)
        # figPseudoRate.yaxis.axis_label = "Range [m]"
        # figPseudoRate.xaxis.axis_label = "Time [s]"
        # figPseudoRate.title.text_font_size = titleFontSize
        # figPseudoRate.xaxis.major_label_text_font_size = tickFontSize
        # figPseudoRate.xaxis.axis_label_text_font_size = axisFontSize
        # figPseudoRate.yaxis.major_label_text_font_size = tickFontSize
        # figPseudoRate.yaxis.axis_label_text_font_size = axisFontSize

        # Pseudorange acceleration
        figPseudoAcc = figure(
            title="Pseudorange Acceleration", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools)
        figPseudoAcc.line(x='time', y='pseudorangeAcc', source=source, line_width=lineWidth)
        figPseudoAcc.yaxis.axis_label = "Range [m]"
        figPseudoAcc.xaxis.axis_label = "Time [s]"
        figPseudoAcc.title.text_font_size = titleFontSize
        figPseudoAcc.xaxis.major_label_text_font_size = tickFontSize
        figPseudoAcc.xaxis.axis_label_text_font_size = axisFontSize
        figPseudoAcc.yaxis.major_label_text_font_size = tickFontSize
        figPseudoAcc.yaxis.axis_label_text_font_size = axisFontSize

        # Doppler
        figDoppler = figure(
            title="Doppler", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools)
        figDoppler.line(x='time', y='doppler', source=source, line_width=lineWidth)
        figDoppler.yaxis.axis_label = "Frequency [Hz]"
        figDoppler.xaxis.axis_label = "Time [s]"
        figDoppler.title.text_font_size = titleFontSize
        figDoppler.xaxis.major_label_text_font_size = tickFontSize
        figDoppler.xaxis.axis_label_text_font_size = axisFontSize
        figDoppler.yaxis.major_label_text_font_size = tickFontSize
        figDoppler.yaxis.axis_label_text_font_size = axisFontSize

        # Doppler noise
        figDopplerNoise = figure(
            title="Doppler Noise", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools)
        figDopplerNoise.line(x='time', y='dopplerNoise', source=source, line_width=lineWidth)
        figDopplerNoise.yaxis.axis_label = "Frequency [Hz]"
        figDopplerNoise.xaxis.axis_label = "Time [s]"
        figDopplerNoise.title.text_font_size = titleFontSize
        figDopplerNoise.xaxis.major_label_text_font_size = tickFontSize
        figDopplerNoise.xaxis.axis_label_text_font_size = axisFontSize
        figDopplerNoise.yaxis.major_label_text_font_size = tickFontSize
        figDopplerNoise.yaxis.axis_label_text_font_size = axisFontSize

        measLayout = layout([[tabTitle],
                              [figPseudorange, figPseudoAcc],
                              [figDoppler, figDopplerNoise]])

        return measLayout
    
    # -------------------------------------------------------------------------

    def _getNavigationTab(self):

        # Map
        mapLayout = self._getMapLayout()

        # Position
        positionLayout = self._getPositionLayout()
        
        layout = pn.Column(mapLayout, positionLayout)

        return layout

    # -------------------------------------------------------------------------

    def _getMapLayout(self):

        # Retrieve from database
        positionList = self.database.fetchPositions()

        refpos = pm.ecef2geodetic( \
            self.referencePosition[0], \
            self.referencePosition[1], \
            self.referencePosition[2], deg=True)

        llh = []
        for position in positionList:
            recpos = position.coordinate.vecpos()
            llh.append(pm.ecef2geodetic(recpos[0], recpos[1], recpos[2], ell=None, deg=True))
        llh = np.array(llh)

        fig = go.Figure(go.Scattermapbox(lat=llh[:,0], lon=llh[:,1], mode='markers'))
        fig.update_layout(
            margin ={'l':0,'t':0,'b':0,'r':0},
            mapbox = {
                'center': {'lon': refpos[1], 'lat': refpos[0]},
                'style': "open-street-map",
                'zoom': 15})

        return fig

    # -------------------------------------------------------------------------

    def _getPositionLayout(self):
        """
        TODO
        """

        # Parameters 
        # TODO Move to config file
        titleFontSize = '16pt'
        tickFontSize = '16pt'
        axisFontSize = '16pt'
        lineWidth = 2

        # Bokeh definitions
        tools = [HoverTool(tooltips=self.tooltips), 'box_select', 'lasso_select', \
            'pan', 'wheel_zoom', 'box_zoom,reset', 'save']

        # Retrieve from database
        positionList = self.database.fetchPositions()

        recpos = []
        recclk = []
        time = []
        timeSample = []
        for position in positionList:
            timeSample.append(position.timeSample / self.rfSignal.samplingFrequency)
            time.append(position.time.datetime)
            recpos.append(position.coordinate.vecpos())
            recclk.append(position.clockError)
        recclk = np.array(recclk)
        
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

        # create a column data source for the plots to share
        # This is to share the lasso selection
        source = ColumnDataSource(data=dict(time=time, timeSample=timeSample, east=enu[:,0], north=enu[:,1], up=enu[:,2], latitude=llh[:,0], longitude=llh[:,1], altitude=llh[:,2]))

        # East/North plot
        height=800
        width=800
        figEN = figure(
            title="East / North", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools)
        figEN.scatter(x='east', y='north', source=source, size=30, marker='dot')
        figEN.scatter(x=np.average(enu[:,0]), y=np.average(enu[:,1]), size=10, fill_color='red')
        figEN.yaxis.axis_label = "North [m]"
        figEN.xaxis.axis_label = "East [m]"
        
        # East
        height=300
        width=1000
        figEast = figure(
            title="East", \
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools,
            y_range=Range1d(-50, 50),
            x_axis_type='datetime')
        figEast.line(x='time', y='east', source=source, line_width=lineWidth)
        figEast.scatter(x='time', y='east', source=source, marker='dot')
        figEast.yaxis.axis_label = "East [m]"
        figEast.xaxis.axis_label = "Time [s]"
        figEast.title.text_font_size = titleFontSize
        figEast.xaxis.major_label_text_font_size = tickFontSize
        figEast.xaxis.axis_label_text_font_size = axisFontSize
        figEast.yaxis.major_label_text_font_size = tickFontSize
        figEast.yaxis.axis_label_text_font_size = axisFontSize

        # North
        figNorth = figure(
            title="North",\
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools,
            y_range=Range1d(-50, 50),
            x_axis_type='datetime')
        figNorth.line(x='time', y='north', source=source, line_width=lineWidth)
        figNorth.scatter(x='time', y='north', source=source, marker='dot')
        figNorth.yaxis.axis_label = "North [m]"
        figNorth.xaxis.axis_label = "Time [s]"
        figNorth.title.text_font_size = titleFontSize
        figNorth.xaxis.major_label_text_font_size = tickFontSize
        figNorth.xaxis.axis_label_text_font_size = axisFontSize
        figNorth.yaxis.major_label_text_font_size = tickFontSize
        figNorth.yaxis.axis_label_text_font_size = axisFontSize

        # Up
        figUp = figure(
            title="Up",\
            background_fill_color=self.backgroundColor,\
            height=height, width=width, tools=tools,
            y_range=Range1d(-50, 50),
            x_axis_type='datetime')
        figUp.line(x='time', y='up', source=source, line_width=lineWidth)
        figUp.scatter(x='time', y='up', source=source, marker='dot')
        figUp.yaxis.axis_label = "Up [m]"
        figUp.xaxis.axis_label = "Time [s]"
        figUp.title.text_font_size = titleFontSize
        figUp.xaxis.major_label_text_font_size = tickFontSize
        figUp.xaxis.axis_label_text_font_size = axisFontSize
        figUp.yaxis.major_label_text_font_size = tickFontSize
        figUp.yaxis.axis_label_text_font_size = axisFontSize

        positionLayout = layout([
                              [figEN],\
                              [figEast],
                              [figNorth],
                              [figUp]])

        # For matplotlib
        #self._plotENU_plt(enu)
        
        return positionLayout

    # -------------------------------------------------------------------------
    
    def _plotENU_plt(self, enu):
        # TODO Move this to a new class
        
        plt.rcParams.update({'font.size': 8})

        fig, ax = plt.subplots(1,1, figsize=(4,4))
        ax.grid(zorder=0)
        ax.set_axisbelow(True)
        plt.plot(enu[:,1], enu[:,0], '+', label='LSE')
        plt.plot(np.average(enu[:,0]), np.average(enu[:,1]), 'o', color='r', label='Average')

        plt.ylabel('North [m]')
        plt.xlabel('East [m]')
        plt.ylim((-30, 30))
        plt.xlim((-30, 30))
        plt.title("East / North errors")
        plt.legend()
        plt.tight_layout()
        plt.savefig('en.png', dpi=300)
        plt.savefig('en.eps')

        print(f'Average ({np.average(enu[:,0]):.2f}m,{np.average(enu[:,1]):.2f}m,{np.average(enu[:,2]):.2f}m) ')
        print(f'STD ({np.std(enu[:,0]):.2f}m,{np.std(enu[:,1]):.2f}m,{np.std(enu[:,2]):.2f}m)')
        print(f'Max ({np.max(np.abs(enu[:,0])):.2f}m,{np.max(np.abs(enu[:,1])):.2f}m,{np.max(np.abs(enu[:,2])):.2f}m)')
        print(f'Min ({np.min(np.abs(enu[:,0])):.2f}m,{np.min(np.abs(enu[:,1])):.2f}m,{np.min(np.abs(enu[:,2])):.2f}m)')
        
        fig, ax = plt.subplots(3,1, figsize=(4,5))
        plt.suptitle("East / North / Up errors")
        ax[0].grid(zorder=0)
        ax[0].set_axisbelow(True)
        ax[0].plot(enu[:,0], label='East')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('East [m]')
        ax[0].set_ylim((-30, 30))

        ax[1].grid(zorder=0)
        ax[1].set_axisbelow(True)
        ax[1].plot(enu[:,1], label='North')
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('North [m]')
        ax[1].set_ylim((-30, 30))

        ax[2].grid(zorder=0)
        ax[2].set_axisbelow(True)
        ax[2].plot(enu[:,2], label='Up')
        ax[2].set_xlabel('Time [s]')
        ax[2].set_ylabel('Up [m]')
        ax[2].set_ylim((-60, 60))

        plt.tight_layout()
        plt.savefig('enu.png', dpi=300)
        plt.savefig('enu_time.eps')

    # -------------------------------------------------------------------------

    def _processingTime(self):

        channelList = self.database.fetchTable('channel')

        for channelID in range(len(channelList)):
            db_entries = self.database.fetchTracking(channelID)

            processTime = []
            for entry in db_entries:
                processTime.append(entry['processTimeNanos'])
            
            processTime = np.array(processTime) / 1e3
            print(f'Channel {channelID}')
            print(f'Average: {np.average(processTime):10.3f} micros')
            print(f'STD    : {np.std(processTime):10.3f} micros')
            print(f'Max    : {np.max(np.abs(processTime)):10.3f} micros')
            print(f'Min    : {np.min(np.abs(processTime)):10.3f} micros')
            print(f'-----')

        return
    
    # -------------------------------------------------------------------------


