
import numpy as np
import pymap3d as pm

class Coordinate():

    x : float
    y : float
    z : float

    vx : float
    vy : float
    vz : float

    xPrecison : float
    yPrecison : float
    zPrecison : float

    # =========================================================================
    # Initialisation functions

    def __init__(self, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0):

        self.x = x
        self.y = y 
        self.z = z

        self.vx = vx 
        self.vy = vy
        self.vz = vz
        
        return

    # -------------------------------------------------------------------------
    
    @staticmethod  
    def fromGeodetic(lat, lon, height):
        x, y, z = pm.geodetic2ecef(lat, lon, height)
        return Coordinate(x, y, z)

    # =========================================================================
    # Setter / Getter
    
    def setCoordinates(self, x, y, z):
        self.x = x
        self.y = y 
        self.z = z
        return

    def setPrecision(self, x, y, z):
        self.xPrecison = x
        self.yPrecison = y 
        self.zPrecison = z
        return
    
    # -------------------------------------------------------------------------

    def vecpos(self):
        return np.array([self.x, self.y, self.z])
    
    # -------------------------------------------------------------------------

    def vecvel(self):
        return np.array([self.vx, self.vy, self.vz])
    
    # =========================================================================
    # Transformations

    def rotate(self, rotation):
        position = rotation.dot(self.vecpos())
        velocity = rotation.dot(self.vecvel())
        self.x, self.y, self.z = list(position)
        self.vx, self.vy, self.vz = list(velocity)
        return
    
    # -------------------------------------------------------------------------

    def getENU(self, referenceCoordinates):
        lat, lon, h = referenceCoordinates.getGeodetic()
        e,n,u = pm.ecef2enu(self.x, self.y, self.z, lat, lon, h)
        return e, n, u

    # -------------------------------------------------------------------------

    def getGeodetic(self, deg=True):
        lat, lon, h = pm.ecef2geodetic(self.x, self.y, self.z, deg=deg)
        lat = float(lat) # Weird fix from the pymap3d toolbox
        return lat, lon, h
    
    # -------------------------------------------------------------------------

    def getAER(self, target, deg=True):
        """
        Get the Azimuth, Elevation, Radial coordinates from this object to the 
        target. 
        """
        lat, lon, h = self.getGeodetic(deg=deg)
        a, e, r = pm.ecef2aer(target.x, target.y, target.z, lat, lon, h, deg=deg)
        return a, e, r

    # =========================================================================

    def __repr__(self):
        return f"x: {self.x:.3f}, y: {self.y:.3f}, z: {self.z:.3f}"

    # -------------------------------------------------------------------------

    