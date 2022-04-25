
from glob import glob
import threading
from gnsstools.channel import Channel

NUMBER_MS = 100

# Number of channel
NB_CHANNELS = 2

# Initialise channels
channels = []
for i in range(NB_CHANNELS):
    channels.append(Channel(i))

threads = []
for i in range(10):
    rfData = i
    
    # Create threads
    threads.clear()
    for idxChannel in range(NB_CHANNELS):
        threads.append(threading.Thread(target=channels[idxChannel].run, args=(rfData, NUMBER_MS)))
    # Start threads
    for thread in threads:
        thread.start()
    # Wait for threads
    for thread in threads:
        thread.join()

