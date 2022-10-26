
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

    # -------------------------------------------------------------------------

    def __init__(self):

        self._resetdX()
        
        return

    # -------------------------------------------------------------------------

    def compute(self):

        N = np.transpose(self.G).dot(self.G)
        C = np.transpose(self.G).dot(self.y)
        dX = np.linalg.inv(N).dot(C)
        
        self.x = self.x + dX             # Update state
        self.v = self.G.dot(dX) - self.y # Update residuals

        return

    # -------------------------------------------------------------------------
    def _resetdX(self):
        self.dX = np.zeros(4)
        self.dX[:4] = [1.0, 1.0, 1.0, 1.0]
        return

    # -------------------------------------------------------------------------
    
    def setDesignMatrix(self, matrix):
        self.G = matrix

    # -------------------------------------------------------------------------

    def setWeigthMatrix(self, matrix):
        self.W = matrix

    # -------------------------------------------------------------------------
    
    def setState(self, position, clock):
        self.x = []
        self.x.extend(position)
        self.x.append(clock)
        return

    # -------------------------------------------------------------------------

    def setObservationVector(self, matrix):
        self.y = matrix

    # -------------------------------------------------------------------------


    
