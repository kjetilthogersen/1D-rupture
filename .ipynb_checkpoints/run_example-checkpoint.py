import numpy as np
import run_continuum as test
import matplotlib.pyplot as plt

x = np.linspace(0,100,1000)
dt = 1e-3
tmax = 30
gamma = 0.65
beta = 0.01
tau = 0.3*np.ones(np.size(x))
tau[500]=1
output_interval = 100

data = test.run_continuum(x = x, tau = tau, tau_minus = tau+2, dt = dt, output_interval = output_interval, gamma = gamma, tmax = tmax, beta = beta)

plt.pcolor(x,data['t'],np.transpose(data['v']))
plt.ylabel('t')
plt.xlabel('x')
plt.title('sliding velocity')
plt.colorbar()

plt.figure()
plt.plot(x,np.gradient(x)/np.gradient(data['unstickTime']))
plt.ylabel('rupture speed')
plt.xlabel('x')

plt.show()