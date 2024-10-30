# %%

import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline


dt      = np.float64(0.05)  # jours
Tinit   = np.float64(0.)
Tmax    = np.float64(20.)  # jours

Nsteps = np.int32( np.ceil( (Tmax-Tinit)/dt ) + 1 ) 
t      = np.linspace(Tinit,Tmax, Nsteps).astype(np.float64) 
y      = 4*t**2






plt.figure(1)
plt.clf()
plt.plot(t, y)

# plt.figure(figsize=(8.27,5.85))
# plt.plot(tvec, S)
# plt.plot(tvec, I)
# plt.plot(tvec, R)
# plt.plot( [tinfl, tinfl] ,[0.,N],'--r')
# plt.plot(tvec[idxForFit], Iinit, ':k')
# plt.ylim((0,N))
# plt.legend(["S(t)","I(t)","R(t)","I$_{max}$","I$_{init}$ $\propto \ exp()$"])
# plt.xlabel("Temps  [jours]");
# plt.ylabel(r"Nombre de cas")
# plt.title(r"S(0)={:.1f}, I(0)={:.2E}, R(0)={:.1f} // Beta={:.2f}, Gamma={:.2f}".format(y0_1, y0_2, y0_3,beta, gamma) )

plt.show()