import numpy as np


# Create pulse prediction function

def PulsePrediction(startPos,u,x,tau,gamma,dc=0):
    u_pred = np.zeros(np.size(x))
    u_pred[startPos]=u[startPos,-1]
    if len([dc])==1:
        dc = dc*np.ones(np.size(x))

    for i in range(startPos+1,len(x)):        
        dx = x[i]-x[i-1]
        u_pred[i] = u_pred[i-1] + (tau[i-1] - .5*gamma*u_pred[i-1])*dx
        
        #Fracture energy mod:
        if u_pred[i-1]<dc[i-1]:
            u_pred[i] = u_pred[i] - (1-u_pred[i-1]/(2*dc[i-1]))*dx
        else:
            u_pred[i] = u_pred[i] - .5*dc[i-1]/(u_pred[i-1])*dx
            
        
            
#        if u_pred[i-1]<dc[i-1]:
#            u_pred[i] = u_pred[i] - (1 - .5*(1-u_pred[i-1]/(2*dc[i-1])))*dx
#        else:
#            u_pred[i] = u_pred[i] - .5*np.sqrt(dc[i-1])/(np.sqrt(u_pred[i-1]))*dx

    
    # Prediction of arrest where u_pred<0
    try:
        ind = np.where(u_pred<0)
        u_pred[ind[0][0]:]=0
    except:
        pass
    
    return u_pred


# Create crack prediction function
def CrackPrediction(u, x, tau, a, left_bc = 'no slip', dc = 0, gamma = 0):

    # 1) Find the arrest based on integral over tau
    dx = x[1]
#    tmp = np.cumsum(tau)*dx
#    L_prediction = x[np.where(tmp<(dc/2))][0]   
    L_prediction = x[np.where( (np.cumsum(tau*dx) - np.cumsum(dc/2*dx)) < 0)][0]
    
#    print(L_prediction)
        
    # 2) Find slip from eom with u=0 at boundaries (see GRL for reference).
    from scipy.sparse import spdiags
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import dsolve
    xP = np.linspace(0,L_prediction,1000)
    dx = xP[1]
    tauP = np.interp(xP,x,tau-a)        
    N_intervals = np.size(xP)-1
    rhs = dx*dx*tauP
    rhs[0] = 0
    rhs[-1] = 0
    A = csc_matrix(spdiags([np.hstack( (-np.ones(N_intervals-1),0,0)) ,  np.hstack(( 1,(2+gamma*dx**2)*np.ones(N_intervals-1),1)),   np.hstack( (0,0,-np.ones(N_intervals-1) ) )],[-1,0,1],N_intervals+1,N_intervals+1))
    u_prediction = dsolve.spsolve(A, rhs, use_umfpack=True)
    u_prediction = np.interp(x,xP,u_prediction)
        
    return u_prediction, L_prediction


#Gauss distribution for setting up initial stress
def gauss(x, mu, sigma):
    return np.exp(-(x - mu)**2/(2*sigma**2))
    