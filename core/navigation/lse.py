
import numpy as np

class LeastSquareEstimation:
    
    # Inputs
    G : np.array # Design matrix
    W : np.array # Weigth matrix
    y : np.array # Observation vector

    # Intermediate
    N : np.array # Normal matrix
    C : np.array # N/A

    # Outputs
    x : np.array # Parameters state
    v : np.array # Residuals

    # Precision
    Qx : np.array # State precision
    Ql : np.array # Measurement precision
    Qv : np.array # Residuals precision

    # -------------------------------------------------------------------------

    def __init__(self):

        self.G = []
        self.W = []
        self.y = []
        self.N = []
        self.C = []
        self.x = []
        self.v = []
        self.Qx = []
        self.Ql = []

        self._resetdX()
        
        return

    # -------------------------------------------------------------------------

    def compute(self):

        N = np.transpose(self.G).dot(self.G)
        C = np.transpose(self.G).dot(self.y)
        dX = np.linalg.inv(N).dot(C)
        
        self.x = self.x + dX             # Update state
        self.v = self.G.dot(dX) - self.y # Update residuals

        self.Qx = np.linalg.inv(N) 
        self.Qv = self.Ql - self.G.dot(self.Qx).dot(np.transpose(self.G))
        self.Ql = self.Ql - self.Qv

        return

    # -------------------------------------------------------------------------
    
    def _resetdX(self):
        self.dX = np.zeros(4)
        self.dX[:4] = [1.0, 1.0, 1.0, 1.0]
        return

    # -------------------------------------------------------------------------

    def setState(self, position, clock):
        self.x = []
        self.x.extend(position)
        self.x.append(clock)
        return

    # -------------------------------------------------------------------------

    def getStatePrecision(self):
        statePrecision = np.sqrt(np.diag(self.Qx))
        return statePrecision



    
