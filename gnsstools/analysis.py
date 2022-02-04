from itertools import accumulate
from blinker import signal
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D 
# from matplotlib import cm
import math
from gnsstools.acquisition import Acquisition

class Analysis:
    def __init__(self):

        self.output_folder = '_results'

        return

    def acquisition(self, acq_list, out_filename, corr_maps_enabled=False):
        NB_COLS = 2

        # Make subplot grid
        grid = [[{'type': 'table'}], [{'type': 'xy'}]]
        # for i in range(0, math.ceil(len(acq_list)/2)):
        #     grid.append([{'type': 'surface'}, {'type': 'surface'}])
        
        # Get subplots title and other stuff
        titles      = [" ", "Acquisition metric"]
        names       = []
        acqMetric  = []
        coarseFreq = []
        coarseDoppler = []
        coarseCode = []
        coarseCodeNorm = []
        for acq in acq_list:
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
        
        if corr_maps_enabled:
            # Loop for correlation results
            i = 0
            for acq in acq_list:
                # Plotting
                x = np.linspace(0, acq.signal.code_bit, np.size(acq.correlationMap, axis=1))
                y = np.arange(-acq.doppler_range, acq.doppler_range, acq.doppler_steps)
                z = acq.correlationMap

                fig_temp = go.Figure(data=[go.Surface(z=z, x=x, y=y,showscale=False)])
                fig_temp.update_layout(title=f"Correlation G{acq.prn} ({acq.signal.name})", autosize=False, \
                    width=1000, height=1000)  
                fig_temp.write_html(f"./{self.output_folder}/Correlation/{out_filename}_G{acq.prn}.html")
            
            fig.write_html(f"./{self.output_folder}/{out_filename}.html")
        
        return



if __name__ == '__main__':
    print("hello")