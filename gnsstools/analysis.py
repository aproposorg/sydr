from itertools import accumulate
from blinker import signal
from matplotlib.pyplot import title
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D 
# from matplotlib import cm
import math
from gnsstools.acquisition import Acquisition
from gnsstools.tracking import Tracking

class Analysis:
    def __init__(self):

        self.output_folder = '_results'

        return

    def acquisition(self, acquisitionList, corrMapsEnabled=False):
        """
        Parse and analyse result from the acquisition process. Graphs are created
        using Plotly and saved to the output folder. 

        Inputs:
        -------
        acquisitionList : List(Acquisition)
            List of Acquisition object.
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
        for prn, acq in acquisitionList.items():
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
            for prn, acq in acquisitionList.items():
                # Plotting
                x = np.linspace(0, acq.signal.code_bit, np.size(acq.correlationMap, axis=1))
                y = np.arange(-acq.doppler_range, acq.doppler_range, acq.doppler_steps)
                z = acq.correlationMap

                fig_temp = go.Figure(data=[go.Surface(z=z, x=x, y=y,showscale=False)])
                fig_temp.update_layout(title=f"Correlation G{acq.prn} ({acq.signal.name})")
                fig_temp.write_html(f"./{self.output_folder}/Correlation/acquisition_G{acq.prn}.html")
            
        
        
        return

    def tracking(self, trackingList):
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
        
        for prn, track in trackingList.items():
            fig = make_subplots(5, 2,\
                start_cell="top-left",
                specs=specs, 
                subplot_titles=titles,
                vertical_spacing=0.1, 
                horizontal_spacing = 0.1)

            time = np.arange(0, track.msProcessed/1e3, 1e-3)

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


if __name__ == '__main__':
    print("hello")