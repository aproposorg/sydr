from signal import signal
import georinex as gr
import matplotlib.pyplot as plt

# =============================================================================

def plotValues(title, obs, satlist, signals, limits):

    fig, axs = plt.subplots(len(signals), figsize=(15,3*len(signals)))
    fig.suptitle(title)

    i = 0
    lines = []
    for sig in signals:
        cn0 = obs[sig].sel(sv=satlist)
        lines.append(axs[i].plot(cn0.time.values, cn0.values))
        axs[i].set_title(sig)
        axs[i].set_ylim(limits[i])
        axs[i].grid()
        
        i += 1

    fig.legend(satlist, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{title}.png')

    return

# =============================================================================

folder = 'c:\\Users\\vmangr\\Documents\\Datasets\\20220615_TAU_Novatel_PolaRx5\\'
rinexObs = 'TAUN00FIN_R_20221660000_01D_30S_MO.22o'

obs = gr.load(folder+rinexObs, tlim=['2022-06-15T00:00', '2022-06-15T23:00'])

# GPS
satlist = []
for sat in obs.sv.values:
    if 'G' in sat:
        satlist.append(sat)
signals = ['S1C', 'S1L', 'S2L', 'S5Q']
cn0limits = [(15, 60), (15, 60), (15, 60), (15, 60)]
plotValues('GPS', obs, satlist, signals, cn0limits)

# Galileo
satlist = []
for sat in obs.sv.values:
    if 'E' in sat:
        satlist.append(sat)
signals = ['S1C', 'S6C', 'S5Q', 'S7Q', 'S8Q']
cn0limits = [(15, 60), (15, 60), (15, 60), (15, 60), (15, 60)]
plotValues('Galileo', obs, satlist, signals, cn0limits)

# BeiDou
satlist = []
for sat in obs.sv.values:
    if 'C' in sat:
        satlist.append(sat)
signals = ['S1P', 'S5P', 'S2I', 'S7I', 'S6I', 'S7D']
cn0limits = [(15, 60), (15, 60), (15, 60), (15, 60), (15, 60), (15, 60)]
plotValues('BeiDou', obs, satlist, signals, cn0limits)



