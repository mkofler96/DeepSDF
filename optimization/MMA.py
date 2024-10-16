    
from mmapy import mmasub
import numpy as np
import logging



class MMA():
    def __init__(self):
        pass
        self.logger =  logging.getLogger(__name__)

    def minimize(self, x0, objective, constraint, bounds, options):
        bounds = np.array(bounds)
        x0 = x0.reshape(-1,1)
        m = 1
        n = len(x0)
        x = x0.copy()
        xold1 = x0.copy()
        xold2 = x0.copy()
        low = []
        upp = []
        a0_MMA = 1
        a_MMA = np.zeros((m,1))
        c_MMA = 10000 * np.ones((m,1))
        d_MMA = np.zeros((m,1))

        loop = 0
        ch = 1.0 
        while(True):
            loop += 1
            C, dC = objective(x)
            Vol, dVol = constraint(x)
            if loop == 1:
                C0 = C
            f0val = C/C0
            df0dx = dC.reshape(-1,1)/C0
            fval = np.array([[Vol]])
            dfdx = dVol.reshape(1,-1)
            
            xmin = np.maximum(x - 0.1, bounds[:,0].reshape(-1,1))
            xmax = np.minimum(x + 0.1, bounds[:,1].reshape(-1,1))
            
            move = 0.1
            # f0val,df0dx,fval,dfdx
            # g0, dg0, g1, dg1
            # from other optimization:
            # f0val.shape
            # (1, 1)
            # df0dx.shape
            # (100, 1)
            # fval
            # 0.0
            # dfdx.shape
            # (1, 100)
            xmma,ymma,zmma,lam,xsi,eta,muMMA,zet,s,low,upp = mmasub(m, n, loop, x, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0_MMA, a_MMA, c_MMA, d_MMA)
            

            xold2 = xold1.copy()
            xold1 = x.copy()
            x = xmma
            ch = np.abs(np.mean(x.T-xold1.T)/np.mean(x.T))

            self.logger.info("It.: {0:4} | Obj.: {1:1.3e} | Constr.:  {2:1.3e} | ch.: {3:1.3e} | C: {4:1.3e}".format(loop, f0val, fval[0][0], ch, C))
            if ch < options["deltaIt"]:
                self.logger.info("Convergence reached")
                break
            if loop == options["maxIt"]:
                self.logger.info("Max Iterations reached")
                break
        return x