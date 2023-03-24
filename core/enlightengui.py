
import enlighten
from termcolor import colored

from core.utils.time import Clock
from core.utils.coordinate import Coordinate
from core.channel.channel import Channel, ChannelStatus
from core.channel.channel_L1CA_2 import ChannelL1CA
from core.dsp.tracking import TrackingFlags

# =====================================================================================================================

STATUS_FORMAT = ' {program}{fill}Stage: {stage}{fill} Status {status} '

RECEIVER_STATUS_FORMAT = u'{receiver} {fill} ' + \
                         u'X: {x:12.4f} (\u03C3: {sx: .4f}) ' + \
                         u'Y: {y:12.4f} (\u03C3: {sy: .4f}) ' + \
                         u'Z: {z:12.4f} (\u03C3: {sz: .4f}) ' + \
                         u'{fill}{datetime} (GPS Time: {gpstime})'

RECEIVER_BAR_FORMAT = u'{desc}{desc_pad}[{state:^10}] {percentage:3.0f}%|{bar}| ' + \
                      u'{count:{len_total}d}/{total:d} ' + \
                      u'[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]'

CHANNEL_BAR_FORMAT = u'{desc} |{prn}| [{state:^10} {tow} SUBFRAMES:{sf1}{sf2}{sf3}{sf4}{sf5}] ' + \
                     u'{percentage:3.0f}%|{bar}| ' + \
                     u'{count:{len_total}d}/{total:d} ' + \
                     u'[{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]'

# =====================================================================================================================

class EnlightenGUI():

    manager : enlighten.Manager

    # GUI components
    main_status_bar       : enlighten.StatusBar
    receiver_status_bar   : enlighten.StatusBar
    receiver_progress_bar : enlighten.Manager.counter
    channels_progress_bars_dict: dict

    # ----------------------------------------------------------------------------------------------------------------

    def __init__(self):
        self.manager = enlighten.Manager()

        # Status bar (bottom)
        self.main_status_bar = self.manager.status_bar(
            status_format=STATUS_FORMAT,
            color='bold_slategray',
            program='SYDR',
            stage='IDLE',
            status='RUNNING',
            position=1)

        return
    
    # ----------------------------------------------------------------------------------------------------------------

    def updateMainStatus(self, stage:str, status:str):

        self.main_status_bar.update(stage=stage, status=status)

        return

    # ----------------------------------------------------------------------------------------------------------------

    def createReceiverGUI(self, receiver):
        """
        """
        
        # Receiver status bar (top)
        clock : Clock = receiver.clock
        coordinate : Coordinate = receiver.coordinate
        self.receiver_status_bar = self.manager.status_bar(
            status_format = RECEIVER_STATUS_FORMAT,
            color         = 'bold_white_on_steelblue4',
            receiver      = receiver.name,
            datetime      = str(receiver.clock.datetime)[:-3],
            gpstime       = f"{clock.getGPSWeek()} {clock.getGPSSeconds():.3f}",
            x             = coordinate.x,
            y             = coordinate.y,
            z             = coordinate.z,
            sx            = coordinate.xPrecison,
            sy            = coordinate.yPrecison,
            sz            = coordinate.zPrecison)
        
        # Receiver processing main progress bar
        self.receiver_progress_bar = self.manager.counter(
            bar_format = RECEIVER_BAR_FORMAT,
            total      = receiver.msToProcess, 
            desc       = f'Processing', 
            unit       = 'ms', 
            color      = 'springgreen3', \
            min_delta  = 0.5, 
            state      = f"{receiver.receiverState}")
        
        # Receiver channels status and progress
        self.channels_progress_bars_dict = {}
        channel : ChannelStatus # Typing for syntax completion 
        for channel in receiver.channelsStatus.values():
            self.channels_progress_bars_dict[channel.channelID] = self.manager.counter(
                bar_format = CHANNEL_BAR_FORMAT, 
                total      = receiver.msToProcess, 
                desc       = f"    Channel {channel.channelID}",
                leave      = False, 
                unit       = 'ms', 
                color      = 'lightseagreen', 
                min_delta  = 0.5,
                state      = f"{channel.channelState}",  
                prn        = f'G{channel.satelliteID:02d}', \
                tow        = colored(" TOW ", 'white', 'on_red'), 
                sf1        = colored("1", 'white', 'on_red'), 
                sf2        = colored("2", 'white', 'on_red'), 
                sf3        = colored("3", 'white', 'on_red'), 
                sf4        = colored("4", 'white', 'on_red'), 
                sf5        = colored("5", 'white', 'on_red'))
        
        return
    
    # ----------------------------------------------------------------------------------------------------------------
    
    def updateReceiverGUI(self, receiver):
        """
        """
        
        # Update receiver status
        clock : Clock = receiver.clock
        coordinate : Coordinate = receiver.coordinate
        self.receiver_status_bar.update(
              datetime = str(receiver.clock.datetime)[:-3], 
              gpstime  = f"{clock.getGPSWeek()} {clock.getGPSSeconds():.3f}",
              x        = coordinate.x,
              y        = coordinate.y,
              z        = coordinate.z,
              sx       = coordinate.xPrecison,
              sy       = coordinate.yPrecison,
              sz       = coordinate.zPrecison)
        
        # Update the receiver progress bar counter
        self.receiver_progress_bar.update(state=f"{receiver.receiverState}")

        # Update channel progress bars
        channel : ChannelStatus # Typing for syntax completion 
        for channel in receiver.channelsStatus.values():
            self.channels_progress_bars_dict[channel.channelID].update(
                state = f"{channel.channelState}", 
                prn   = f'G{channel.satelliteID:02d}', 
                tow   = colored(f" TOW: {channel.tow:6.0f}", 'white', 'on_green' if (channel.trackFlags & TrackingFlags.TOW_DECODED) else 'on_red'),
                sf1   = colored("1", 'white', 'on_green' if channel.subframeFlags[0] else 'on_red'),
                sf2   = colored("2", 'white', 'on_green' if channel.subframeFlags[1] else 'on_red'),
                sf3   = colored("3", 'white', 'on_green' if channel.subframeFlags[2] else 'on_red'),
                sf4   = colored("4", 'white', 'on_green' if channel.subframeFlags[3] else 'on_red'),
                sf5   = colored("5", 'white', 'on_green' if channel.subframeFlags[4] else 'on_red'))
        
        return
    
    # -----------------------------------------------------------------------------------------------------------------