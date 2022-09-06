from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import numpy as np
import os
from core.acquisition.acquisition_abstract import AcquisitionAbstract as Acquisition
from core.navigation import Navigation
import core.utils.constants as constants
import pymap3d as pm

from core.signal.rfsignal import RFSignal

class Analysis:
    def __init__(self, rfConfig:RFSignal):

        self.rfConfig = rfConfig
        self.output_folder = '_results'

        return

    def acquisition(self, satelliteDict, corrMapsEnabled=False):
        """
        Parse and analyse result from the acquisition process. Graphs are created
        using Plotly and saved to the output folder. 

        Args:
            satelliteDict (Dictionary): Dictionnary of results.
        corrMapsEnabled : Boolean
            Enable plotting and saving of correlation maps for each satellites.

        Outputs:
        --------
        None
        """
        # Make subplot grid
        grid = [[{'type': 'table'}], [{'type': 'xy'}]]
        
        # Get subplots title and other stuff
        titles      = [" ", "Acquisition metric"]
        names       = []
        acqMetric  = []
        coarseFreq = []
        coarseDoppler = []
        coarseCode = []
        coarseCodeNorm = []
        threshold = []
        for prn, results in satelliteDict.items():
            dsp = results.dspMeasurements
            # TODO change
            dsp = dsp[0]
            signalConfig = dsp.signalConfig
            samplesPerCode = round(self.rfConfig.samplingFrequency / (signalConfig.codeFrequency / signalConfig.codeBits))
            titles.append(f"G{prn} ({dsp.signalConfig.signalType})")
            names.append(f"G{prn} ({dsp.signalConfig.signalType})")
            acqMetric.append(f"{dsp.acquisitionMetric:>6.2f}")
            coarseFreq.append(f"{dsp.estimatedFrequency:>8.2f}")    
            coarseDoppler.append(f"{(dsp.estimatedFrequency-self.rfConfig.interFrequency):>8.2f}")   
            coarseCode.append(f"{dsp.estimatedCode:>8.2f}")
            coarseCodeNorm.append(f"{dsp.estimatedCode * signalConfig.codeBits / samplesPerCode:>8.2f}")
            threshold.append(results.acquisition[0].metricThreshold)
        
        # Make de subplot
        fig = make_subplots(2, 1,\
            start_cell="top-left",
            specs=grid, 
            subplot_titles=titles, 
            vertical_spacing=0.05, 
            horizontal_spacing = 0.05)

        # Results table
        fig.add_trace(go.Table(
            header=dict(
                    values=["PRN", "Metric", "Frequency [Hz]", "Doppler [Hz]", "Code phase", "Code phase (normalised)"],
                    font=dict(size=12),
                    align="left"
            ),
            cells=dict(
                values=[names, acqMetric, coarseFreq, coarseDoppler, coarseCode, coarseCodeNorm],
                align = "right")
            ),
            row=1, col=1)
        
        # Results bar chart
        fig.add_trace(go.Bar(x=names, y=[float(i) for i in acqMetric], width=800, height=400), row=2, col=1)
        fig.add_trace(go.Scatter(x=names, y=threshold, line=dict(width=2), width=800, height=400),row=2, col=1)
        fig.update_layout(title=f"Acquisition", showlegend=False)
        fig.write_html(f"./{self.output_folder}/acquisition.html")
        
        if corrMapsEnabled:
            for prn, results in satelliteDict.items():
                i = 0
                for acq in results.acquisition:
                    path =  f"./{self.output_folder}/correlation/acquisition_{i}"
                    self.plotCorrelation(acq, path)
                    i += 1
        
        return

    def plotCorrelation(self, acquisition:Acquisition, path):
        codeSpace = np.linspace(0, acquisition.signalConfig.codeBits, np.size(acquisition.correlationMap, axis=1))
        frequencySpace = acquisition.frequencyBins
        z = acquisition.correlationMap

        idxFrequency = acquisition.idxEstimatedFrequency
        idxCode = acquisition.idxEstimatedCode

        grid   = [[{'type': 'xy'}, {'type': 'xy'}], 
                  [{'type': 'surface', "colspan": 2}, None]]
        titles = ["Frequency correlation", "Code correlation", "Correlation map"]

        fig = make_subplots(2, 2,\
            start_cell="top-left",
            specs=grid, 
            subplot_titles=titles, 
            vertical_spacing=0.05, 
            horizontal_spacing = 0.05)

        # Frequency correlation
        pos = (1,1)
        fig.add_trace(go.Scatter(x=frequencySpace, y=acquisition.correlationMap[:, idxCode],
                            	line=dict(width=2)), row=pos[0], col=pos[1])
        fig.update_xaxes(title_text="Frequency [Hz]", row=pos[0], col=pos[1], \
            showgrid=True, gridcolor='LightGray')
        fig.update_yaxes(title_text="Amplitude", row=pos[0], col=pos[1], \
            showgrid=True, gridcolor='LightGray')

        # Code correlation
        pos = (1,2)
        fig.add_trace(go.Scatter(x=codeSpace, y=acquisition.correlationMap[idxFrequency, :],
                            	line=dict(width=2)), row=pos[0], col=pos[1])
        fig.update_xaxes(title_text="Samples", row=pos[0], col=pos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray', constrain="domain")
        fig.update_yaxes(title_text="Amplitude", row=pos[0], col=pos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # Correlation map
        pos = (2,1)
        fig.add_trace(go.Surface(z=z, x=codeSpace, y=frequencySpace, showscale=False), \
            row=pos[0], col=pos[1])
        
        fig.update_layout(title=f"Correlation G{acquisition.svid} ({acquisition.signalConfig.signalType})", showlegend=False)
        fig.write_html(f"{path}_G{acquisition.svid}.html")

        return

    def tracking(self, satelliteDict):
        # Make subplot grid
        specs = [[{'type': 'xy'}, None],
                 [{'type': 'xy'}, {'type': 'xy'}], 
                 [{'type': 'xy'}, {'type': 'xy'}], 
                 [{'type': 'xy',"colspan": 2}, None],
                 [{'type': 'xy',"colspan": 2}, None]]

        titles = ["I/Q decorrelation (last second only)", 
                  "Raw DLL discriminator (Relative)", "Raw DLL discriminator (Absolute)",
                  "Raw PLL discriminator (Relative)", "Raw PLL discriminator (Absolute)",
                  "In-phase (I) Prompt", 
                  "Quadraphase (Q) Prompt"]
        
        for prn, results in satelliteDict.items():
            dsp = results.dspMeasurements
            fig = make_subplots(5, 2,\
                start_cell="top-left",
                specs=specs, 
                subplot_titles=titles,
                vertical_spacing=0.1, 
                horizontal_spacing = 0.1)

            # TODO change
            dsp = dsp[0]

            time = np.arange(0, len(dsp.iPrompt)/1e3, 1e-3)

            colors = plotly.colors.DEFAULT_PLOTLY_COLORS
            
            # I/Q scatter
            last = 1000
            pos = (1,1)
            fig.add_trace(go.Scatter(x=dsp.iPrompt[-last:], y=dsp.qPrompt[-last:], mode="markers",
                            	    line=dict(width=2, color=colors[0])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray', constrain="domain")
            fig.update_yaxes(title_text="Amplitude", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray', scaleanchor="x", scaleratio=1)

            # Raw DLL discriminator (Relative)
            pos = (2,1)
            fig.add_trace(go.Scatter(x=time, y=dsp.dll,
                            	    line=dict(width=2, color=colors[0])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text="Amplitude", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')

            # Raw DLL discriminator (Absolute)
            pos = (2,2)
            fig.add_trace(go.Scatter(x=time, y=dsp.codeFrequency,
                            	    line=dict(width=2, color=colors[0])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text="Amplitude", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')

            # Raw PLL discriminator (Relative)
            pos = (3,1)
            fig.add_trace(go.Scatter(x=time, y=dsp.pll,
                            	    line=dict(width=2, color=colors[1])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text="Frequency [Hz]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            # Raw PLL discriminator (Absolute)
            pos = (3,2)
            fig.add_trace(go.Scatter(x=time, y=dsp.carrierFrequency,
                            	    line=dict(width=2, color=colors[1])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text="Frequency Hz]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')

            # In-phase (I) Prompt
            pos = (4,1)
            fig.add_trace(go.Scatter(x=time, y=dsp.iPrompt,
                            	    line=dict(width=2, color=colors[2])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text="Amplitude", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')

            # Quadraphase (Q) Prompt
            pos = (5,1)
            fig.add_trace(go.Scatter(x=time, y=dsp.qPrompt,
                            	    line=dict(width=2, color=colors[3])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text="Amplitude", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')

            fig.update_layout(title=f"Tracking G{prn} ({dsp.signalConfig.signalType})", showlegend=False) 
            fig.write_html(f"./{self.output_folder}/tracking_{prn}.html")
        

        return

    def navigation(self, navigationResults:Navigation):

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
        
        recpos = np.array(navigationResults.receiverPosition)
        recclk = np.array(navigationResults.receiverClockError)
        refpos = np.array(navigationResults.referenceReceiverPosition)
        
        # Convert to ENU
        enu = []
        llh = []
        for coord in recpos:
            enu.append(pm.ecef2enu(coord[0], coord[1], coord[2], refpos[0], refpos[1], refpos[2]))
            llh.append(pm.ecef2geodetic(coord[0], coord[1], coord[2], ell=None, deg=True))
        enu = np.array(enu)
        llh = np.array(llh)

        time = np.linspace(0, navigationResults.msToProcess/1e3, len(recpos))

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
        fig.add_trace(go.Scatter(x=time, y=recclk/constants.SPEED_OF_LIGHT, mode="lines+markers",
                                line=dict(width=2)), row=figpos[0], col=figpos[1])
        fig.update_xaxes(title_text="Time [s]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(title_text="Bias [s]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')

        fig.update_layout(title=f"Navigation solution", margin=dict(l=50,r=50,b=100,t=100,pad=4),showlegend=False) 
        fig.write_html(f"./{self.output_folder}/navigation.html")

        return

    def navComputations(self, navigationResults:Navigation):

        return

    def measurements(self, satelliteDict):
        specs = [[{'type': 'table', "colspan": 2}, None],
                 [{'type': 'xy'}, {'type': 'xy'}], 
                 [{'type': 'xy'}, {'type': 'xy'}]]

        titles = ["Table of measurements", 
                  "Pseudorange", "Pseudorange noise",
                  "Doppler", "Doppler noise"]
        
        for prn, results in satelliteDict.items():
            
            n = len(results.pseudoranges)
            time = np.array(results.measurementsTOW) - results.measurementsTOW[0]

            fig = make_subplots(3, 2,\
                start_cell="top-left",
                specs=specs, 
                subplot_titles=titles,
                vertical_spacing=0.1, 
                horizontal_spacing = 0.1)

            colors = plotly.colors.DEFAULT_PLOTLY_COLORS
            
            # Measurement table
            pos = (1,1)
            fig.add_trace(go.Table(
            header=dict(
                    values=["Time [s]", "TOW [s]", "Pseudorange [m]", "Doppler [Hz]", "Carrier phase [Cycle]", "C/N0"],
                    font=dict(size=12),
                    align="left"
            ),
            cells=dict(
                values=[time, results.measurementsTOW, results.pseudoranges, results.doppler, np.zeros(n), np.zeros(n)],
                align = "right", format=(".1f", ".1f", "8.3f", "8.3f", "8.3f", "8.3f"))
            ),
            row=pos[0], col=pos[1])

            # Pseudorange
            pos = (2,1)
            fig.add_trace(go.Scatter(x=time, y=results.pseudoranges,
                                     line=dict(width=2, color=colors[0])),
                            row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray', exponentformat="none")
            fig.update_yaxes(title_text="Range [m]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray', exponentformat="e")

            # Pseudorange error
            pos = (2,2)
            diff = np.array([results.pseudoranges[i] - results.pseudoranges[i-1] for i in range(1,n)])
            error = np.array([diff[i] - diff[i-1] for i in range(1,len(diff))])
            fig.add_trace(go.Scatter(x=time, y=error, 
                                     line=dict(width=2, color=colors[0])),
                            row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray', exponentformat="none")
            fig.update_yaxes(title_text="Error [m]", range=(-50, 50), row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')

            # Doppler 
            pos = (3,1)
            fig.add_trace(go.Scatter(x=time, y=results.doppler, 
                                     line=dict(width=2, color=colors[1])),
                            row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray', exponentformat="none")
            fig.update_yaxes(title_text="Frequency shift [Hz]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray', exponentformat="e")

            # Doppler error
            pos = (3,2)
            #diff = np.array([results.doppler[i] - results.doppler[i-1] for i in range(1,len(results.coarsePseudoranges))])
            error = np.array([results.doppler[i] - results.doppler[i-1] for i in range(1,len(results.doppler))])
            fig.add_trace(go.Scatter(x=time, y=error,
                                     line=dict(width=2, color=colors[1])),
                            row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray', exponentformat="none")
            fig.update_yaxes(title_text="Error [Hz]", range=(-10, 10), row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')

            fig.update_layout(title=f"Measurements G{results.tracking.prn} ({results.tracking.signal.name})", showlegend=False) 
            fig.write_html(f"./{self.output_folder}/measurements_{results.tracking.prn}.html")
            
            
        return


    def analyseDSP(self, satellites):

        for prn, sat in satellites.items():
            grid = [[{'type': 'table'}, {'type': 'xy'}, {'type': 'xy'}], [{'type': 'xy'}]]



        return