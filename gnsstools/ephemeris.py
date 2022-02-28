import numpy as np

class Ephemeris:
    def __init__(self):
        self.weekNumber = np.NaN
        self.accuracy   = np.NaN
        self.health     = np.NaN
        self.IODC       = np.NaN
        self.t_oc       = np.NaN
        self.T_GD       = np.NaN
        self.a_f2       = np.NaN
        self.a_f1       = np.NaN
        self.a_f0       = np.NaN
        self.IODE_sf2   = np.NaN
        self.e          = np.NaN
        self.sqrtA      = np.NaN
        self.t_oe       = np.NaN
        self.C_rs       = np.NaN
        self.deltan     = np.NaN
        self.M_0        = np.NaN
        self.C_uc       = np.NaN
        self.C_us       = np.NaN
        self.C_ic       = np.NaN
        self.omega_0    = np.NaN
        self.C_is       = np.NaN
        self.i_0        = np.NaN
        self.C_rc       = np.NaN
        self.omega      = np.NaN
        self.omegaDot   = np.NaN
        self.iDot       = np.NaN
        self.IODE_sf3   = np.NaN