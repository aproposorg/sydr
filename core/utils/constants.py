
# =====================================================================================================================
# Generic constants
PI                 = 3.1415926535898 # GPS' definition of Pi  
HALF_PI            = PI / 2.0
TWO_PI             = PI * 2.0
SPEED_OF_LIGHT     = 299792458.0 

# =====================================================================================================================
# GNSS TIME AND GEODESY
# Time 
SECONDS_PER_DAY = 86400.0
GPS_WEEK_ROLLOVER = 2

# Pseudorange constants
AVG_TRAVEL_TIME_MS = 76.0

# Earth constants
EARTH_RADIUS        =  6378137.0       # [m]
EARTH_ROTATION_RATE =  7.2921151467e-5 # [ras/s] Also known as Omega dot
#EARTH_GM            =  3.986004418e14  # [m^3/s^2] Also known as the Standard Gravitational constant mu
EARTH_GM            = 3.986005e14  # [m^3/s^2] Also known as the Standard Gravitational constant mu

# Orbit constants
RELATIVIST_CLOCK_F  = -4.442807633e-10 # [s/m^(1/2)] Relativist clock correction

# Atmospheric corrections
# From [ESA, 2013]
IONO_MAG_LAT = 78.3  # [degrees] Geomagnetic pole latitude
IONO_MAG_LON = 291.0 # [degrees] Geomagnetic pole longitude

TROPO_K1  = 77.604   # [K/mbar]
TROPO_K2  = 382000.0 # [K^2/mbar]
TROPO_R   = 287.054  # [J/(Kg K)]
TROPO_G   = 9.80665  # [m/s^2]
TROPO_G_M = 9.784    # [m/s^2]

# Standard values for meteorological parameters 
TROPO_METEO_AVG_LAT = [  15.00,   30.00,   45.00,   60.00,   75.00] # [Degrees]
TROPO_METEO_AVG_P0  = [1013.25, 1017.25, 1015.75, 1011.75, 1013.00] # [mbar]
TROPO_METEO_AVG_T0  = [ 299.65,  294.15,  283.15,  272.15,  263.65] # [K]
TROPO_METEO_AVG_E0  = [  26.31,   21.79,   11.66,    6.78,    4.11] # [mbar]
TROPO_METEO_AVG_B0  = [6.30e-3, 6.05e-3, 5.58e-3, 5.39e-3, 4.53e-3] # [K/m]
TROPO_METEO_AVG_L0  = [   2.77,    3.15,    2.57,    1.81,    1.55]

TROPO_METEO_VAR_LAT = [  15.00,   30.00,   45.00,   60.00,   75.00] # [Degrees]
TROPO_METEO_VAR_P0  = [    0.0,   -3.75,   -2.25,   -1.75,   -0.50] # [mbar]
TROPO_METEO_VAR_T0  = [    0.0,    7.00,   11.00,   15.00,   14.50] # [K]
TROPO_METEO_VAR_E0  = [    0.0,    8.85,    7.24,    5.36,    3.39] # [mbar]
TROPO_METEO_VAR_B0  = [    0.0, 0.25e-3, 0.32e-3, 0.81e-3, 0.62e-3] # [K/m]
TROPO_METEO_VAR_L0  = [    0.0,    0.33,    0.46,    0.74,    0.30]

# DISPLAY
UNI_SIGMA = '\u03C3'

# =====================================================================================================================
# NAVIGATION MESSAGES

# LNAV (GPS)
LNAV_PREAMBULE_BITS     = [1, 0, 0, 0, 1, 0, 1, 1] # Preambule bit sequence
LNAV_PREAMBULE_BITS_INV = [0, 1, 1, 1, 0, 1, 0, 0] # Inverse preambule bit sequence
LNAV_PREAMBULE_SIZE = 8   # Number of bits in preambule
LNAV_MS_PER_BIT     = 20  # Number of milliseconds per navigation bits
LNAV_SUBFRAME_SIZE  = 300 # Number of bits per subframe
LNAV_WORD_SIZE      = 30  # Number of bits per word


# =====================================================================================================================
# GNSS SIGNALS

GPS_L1CA_NAME = "GPS L1 C/A"       # String representation
GPS_L1CA_CARRIER_FREQ = 1575.42e6  # [Hz] Carrier frequency in [MHz]
GPS_L1CA_CODE_SIZE_BITS = 1023     # Number bit per C/A code    
GPS_L1CA_CODE_FREQ = 1.023e6       # [Hz] C/A code frequency 
GPS_L1CA_CODE_MS = 1               # Number of code per millisecond of signal

# =====================================================================================================================
# DIGITAL LOOP FILTERS

W0_BANDWIDTH_1     = 0.25   # Digital Loop filter (DLF) scale constant. See [Kaplan, 2006], p180.
W0_BANDWIDTH_2     = 0.53   # Digital Loop filter (DLF) scale constant. See [Kaplan, 2006], p180.
W0_BANDWIDTH_3     = 0.7845 # Digital Loop filter (DLF) scale constant. See [Kaplan, 2006], p180.
W0_SCALE_A2        = 1.414  # 2nd order DLF scale constant. See [Kaplan, 2006], p180.
W0_SCALE_A3        = 1.1    # 3rd order DLF scale constant. See [Kaplan, 2006], p180.
W0_SCALE_B3        = 2.4    # 3rd order DLF scale constant. See [Kaplan, 2006], p180.
