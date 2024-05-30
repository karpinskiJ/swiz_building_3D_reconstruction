import numpy as np

class Camera():
    def __init__(self,P = None) -> None:
        self.P = P
    
    def _factorization(self):
        if self.P:
            Q, R = np.linalg.qr(np.flipud(self.P).T)
            R = np.flipud(R.T)
            K,R =  R[:, ::-1], Q.T[::-1, :]
            T = np.diag(np.sign(np.diag(K)))

            if np.linalg.det(T) < 0:
                T[1, 1] *= -1

            self.K = np.dot(K, T)
            self.R = np.dot(T, R)
            self.t = np.dot(np.linalg.inv(self.K), self.P[:, 3])

    def getK(self):
        return self.K

    def getR(self):
        return self.R
    
    def getT(self):
        return self.t
    
    def getP(self):
        return self.P
    
if __name__ == '__main__':
    pass
