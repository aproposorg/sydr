from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import numpy as np
from gnsstools.navigation import Navigation
import pymap3d as pm

class Analysis:
    def __init__(self):

        self.output_folder = '_results'

        return

    def acquisition(self, satelliteDict, corrMapsEnabled=False):
        """
        Parse and analyse result from the acquisition process. Graphs are created
        using Plotly and saved to the output folder. 

        Inputs:
        -------
        satelliteDict : Dict()
            Dictionnary of results.
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
        for prn, results in satelliteDict.items():
            acq = results.getAcquisition()
            titles.append(f"G{acq.prn} ({acq.signal.name})")
            names.append(f"G{acq.prn} ({acq.signal.name})")
            acqMetric.append(f"{acq.acqMetric:>6.2f}")
            coarseFreq.append(f"{acq.coarseFreq:>8.2f}")    
            coarseDoppler.append(f"{acq.coarseDoppler:>8.2f}")   
            coarseCode.append(f"{acq.coarseCode:>8.2f}")
            coarseCodeNorm.append(f"{acq.coarseCodeNorm:>8.2f}")
        
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
        fig.add_trace(go.Bar(x=names, y=[float(i) for i in acqMetric]), row=2, col=1)
        fig.write_html(f"./{self.output_folder}/acquisition.html")
        
        if corrMapsEnabled:
            # Loop for correlation results
            i = 0
            for prn, results in satelliteDict.items():
                acq = results.getAcquisition()
                x = np.linspace(0, acq.signal.code_bit, np.size(acq.correlationMap, axis=1))
                y = np.arange(-acq.doppler_range, acq.doppler_range, acq.doppler_steps)
                z = acq.correlationMap

                fig_temp = go.Figure(data=[go.Surface(z=z, x=x, y=y,showscale=False)])
                fig_temp.update_layout(title=f"Correlation G{acq.prn} ({acq.signal.name})")
                fig_temp.write_html(f"./{self.output_folder}/Correlation/acquisition_G{acq.prn}.html")
            
        
        
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
            track = results.getTracking()
            fig = make_subplots(5, 2,\
                start_cell="top-left",
                specs=specs, 
                subplot_titles=titles,
                vertical_spacing=0.1, 
                horizontal_spacing = 0.1)

            time = np.arange(0, len(track.iPrompt)/1e3, 1e-3)

            colors = plotly.colors.DEFAULT_PLOTLY_COLORS
            
            # I/Q scatter
            last = 1000
            pos = (1,1)
            fig.add_trace(go.Scatter(x=track.iPrompt[-last:], y=track.qPrompt[-last:], mode="markers",
                            	    line=dict(width=2, color=colors[0])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray', constrain="domain")
            fig.update_yaxes(title_text="Amplitude", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray', scaleanchor="x", scaleratio=1)

            # Raw DLL discriminator (Relative)
            pos = (2,1)
            fig.add_trace(go.Scatter(x=time, y=track.codeNCO,
                            	    line=dict(width=2, color=colors[0])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text="Amplitude", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')

            # Raw DLL discriminator (Absolute)
            pos = (2,2)
            fig.add_trace(go.Scatter(x=time, y=track.codeFrequency,
                            	    line=dict(width=2, color=colors[0])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text="Amplitude", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')

            # Raw PLL discriminator (Relative)
            pos = (3,1)
            fig.add_trace(go.Scatter(x=time, y=track.carrierNCO,
                            	    line=dict(width=2, color=colors[1])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text="Frequency [Hz]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            # Raw PLL discriminator (Absolute)
            pos = (3,2)
            fig.add_trace(go.Scatter(x=time, y=track.carrierFrequency,
                            	    line=dict(width=2, color=colors[1])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text="Frequency Hz]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')

            # In-phase (I) Prompt
            pos = (4,1)
            fig.add_trace(go.Scatter(x=time, y=track.iPrompt,
                            	    line=dict(width=2, color=colors[2])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text="Amplitude", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')

            # Quadraphase (Q) Prompt
            pos = (5,1)
            fig.add_trace(go.Scatter(x=time, y=track.qPrompt,
                            	    line=dict(width=2, color=colors[3])), row=pos[0], col=pos[1])
            fig.update_xaxes(title_text="Time [s]", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(title_text="Amplitude", row=pos[0], col=pos[1], \
                showgrid=True, gridwidth=1, gridcolor='LightGray')

            fig.update_layout(title=f"Tracking G{track.prn} ({track.signal.name})") 
            fig.write_html(f"./{self.output_folder}/tracking_{track.prn}.html")
        

        return

    def navigation(self, navigationResults:Navigation):

        specs = [[{'type': 'mapbox'}, {'type': 'xy'}],
                 [{'type': 'xy',"colspan": 2}, None],
                 [{'type': 'xy',"colspan": 2}, None],
                 [{'type': 'xy',"colspan": 2}, None],
                 [{'type': 'xy',"colspan": 2}, None]]

        titles = ["Map", "North/East",
                  "North [m]",
                  "East [m]",
                  "Up [m]",
                  "Recevier Clock error [m]"]
        
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

        fig = make_subplots(5, 2,\
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
                'zoom': 13})

        # North/East view
        figpos = (1,2)
        fig.add_trace(go.Scatter(x=enu[:,0], y=enu[:,1], mode="markers",
                                line=dict(width=2)), row=figpos[0], col=figpos[1])
        fig.update_xaxes(title_text="East [m]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(title_text="North [m]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray', scaleanchor="x", scaleratio=1)

        # East
        figpos = (2,1)
        fig.add_trace(go.Scatter(x=time, y=enu[:,0], mode="lines+markers",
                                line=dict(width=2)), row=figpos[0], col=figpos[1])
        fig.update_xaxes(title_text="Time [s]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(title_text="East [m]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # North
        figpos = (3,1)
        fig.add_trace(go.Scatter(x=time, y=enu[:,1], mode="lines+markers",
                                line=dict(width=2)), row=figpos[0], col=figpos[1])
        fig.update_xaxes(title_text="Time [s]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(title_text="North [m]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # Up
        figpos = (4,1)
        fig.add_trace(go.Scatter(x=time, y=enu[:,2], mode="lines+markers",
                                line=dict(width=2)), row=figpos[0], col=figpos[1])
        fig.update_xaxes(title_text="Time [s]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(title_text="Up [m]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')

        # Receiver time error
        figpos = (5,1)
        fig.add_trace(go.Scatter(x=time, y=recclk, mode="lines+markers",
                                line=dict(width=2)), row=figpos[0], col=figpos[1])
        fig.update_xaxes(title_text="Time [s]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(title_text="Up [m]", row=figpos[0], col=figpos[1], \
            showgrid=True, gridwidth=1, gridcolor='LightGray')

        fig.update_layout(title=f"Navigation solution") 
        fig.write_html(f"./{self.output_folder}/navigation.html")

        return