#
# Solves 1D elasticty with Amontons-Coulomb friction in non-dimensional form given the prestress tau, damping beta, position x, elasticity ratio gamma.
#
#


import numpy as np
def run_continuum(x, tau, tau_minus, dt, gamma, output_interval = 1, tmax = 1, beta = 0.01, bc = 'fixed', frictionLaw = None):

	N = len(x)
	u = np.zeros(np.size(x))
	v = np.zeros(np.size(x))
	a = np.zeros(np.size(x))
	stuck = np.ones(np.size(x))*True

	Nsave = np.min([int(np.ceil(np.max(x)*10/(dt*output_interval))), int(tmax/dt/output_interval+1) ]) #Initalize arrays with size based on the system size and a rupture speed of 1/10 v_s or the maximum time if given.

	u_out = np.zeros((N,Nsave))
	v_out = np.zeros((N,Nsave))
	a_out = np.zeros((N,Nsave))
	stuck_out = np.zeros((N,Nsave))
	t_out = np.zeros(Nsave)

	unstickTime = np.full((N),np.nan)
	#stickTime = np.full((N),np.nan)

	i = 0
	output_ind = 0
	t = 0
	while t<tmax:
		v_prev = np.copy(v)
		stuck_prev = np.copy(stuck)
		taubar = np.copy(tau)
		taubar[v<0]=tau_minus[v<0] # In case of negative velocities

		if bc=='fixed':
			utmp = np.insert(u,(0,len(u)),(0,0))
			vtmp = np.insert(v,(0,len(v)),(0,0))
		elif bc=='force_left':
			utmp = np.insert(u,(0,len(u)),(u[0],0))
			vtmp = np.insert(v,(0,len(v)),(v[0],v[-1]))

		xtmp = np.insert(x,(0,len(x)),(2*x[0]-x[1], 2*x[-1]-x[-2] ))
		dx = np.diff(xtmp)

		# accelaration
		a = -gamma*u + taubar + ((utmp[2:]-utmp[1:-1])/dx[0:-1] - (utmp[1:-1]-utmp[0:-2])/dx[1:])/((dx[0:-1]+dx[1:])/2) + beta*((vtmp[2:]-vtmp[1:-1])/dx[0:-1] - (vtmp[1:-1]-vtmp[0:-2])/dx[1:])/((dx[0:-1]+dx[1:])/2)

        # add friction law (supplied as a frictionLaw object)
		if frictionLaw is not None:
			frictionMod = frictionLaw.getFriction(x,u,v,tau,stuck)
			a[stuck==False] = a[stuck==False] - frictionMod[stuck==False]

		#Store first time of ruputure (for more precise rupture speed than output allows for)
		unstickTime[ ((a>=1)|(a<-1)) & (stuck==True) & (unstickTime is not float('nan')) ] = t

		# Unstick if static threshold surpassed
		stuck[a>=1]=False
		stuck[a<=-1]=False

		#Euler Cromer step
		v[stuck==False] = v[stuck==False] + a[stuck==False]*dt
		stuck[((v<0)&(v_prev>0)) & (stuck_prev==False)]=True #Stick if velocity changes sign
		v[stuck==True]=0
		u[stuck==False] = u[stuck==False] + v[stuck==False]*dt
		if frictionLaw is not None:
			frictionLaw.step(x,u,v,tau,stuck,dt)

		if np.sum(stuck)==N: # Stop simulation if all blocks are stuck (and store final step).
			u_out[:,output_ind+1] = u
			v_out[:,output_ind+1] = v
			a_out[:,output_ind+1] = a
			stuck_out[:,output_ind+1] = stuck
			t_out[output_ind+1] = t
			break

		i+=1
		t = dt*i

		if i%output_interval==0: # Prepare output
			u_out[:,output_ind] = u
			v_out[:,output_ind] = v
			a_out[:,output_ind] = a
			stuck_out[:,output_ind] = stuck

			t_out[output_ind] = t
			output_ind+=1


	#Place output in dictionary:
	out = {'x': x,
		'tau': tau,
		'u': u_out[:,0:output_ind],
		'v': v_out[:,0:output_ind],
		'a': a_out[:,0:output_ind],
		'stuck': stuck_out[:,0:output_ind],
		't': t_out[0:output_ind],
		'unstickTime': unstickTime,
		'beta': beta,
		'gamma': gamma,
		'frictionLaw': frictionLaw
	}

	return out
