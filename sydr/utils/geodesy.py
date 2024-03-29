
import numpy as np

from sydr.utils.constants import EARTH_ROTATION_RATE

# =====================================================================================================================

def correctEarthRotation(traveltime, X_sat):
    """
    E_R_CORR  Returns rotated satellite ECEF coordinates due to Earth
    rotation during signal travel time

    X_sat_rot = e_r_corr(traveltime, X_sat);

        Inputs:
            travelTime  - signal travel time
            X_sat       - satellite's ECEF coordinates

        Outputs:
            X_sat_rot   - rotated satellite's coordinates (ECEF)

    Written by Kai Borre
    Copyright (c) by Kai Borre
    """

    # Find rotation angle 
    omegatau = EARTH_ROTATION_RATE * traveltime

    # Make a rotation matrix 
    R3 = np.array([[np.cos(omegatau), np.sin(omegatau), 0.0],
                   [-np.sin(omegatau), np.cos(omegatau), 0.0],
                   [0.0, 0.0, 1.0]])

    #  Do the rotation 
    X_sat_rot = R3.dot(X_sat)
    
    return X_sat_rot

# =====================================================================================================================