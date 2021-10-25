import numpy as np

# Friction classes for alternative friction laws:
class SlipWeakeningFrictionLaw: 
    
    def __init__(self,dc,delta_u):
        self.dc = dc
        self.delta_u = delta_u
        
    def getFriction(self,x,u,v,tau,stuck):
        frictionForce = np.sign(v)*(1-self.delta_u/self.dc)
        frictionForce[frictionForce<0]=0
        return frictionForce
    
    def step(self,x,u,v,tau,stuck,dt):
        self.delta_u = self.delta_u + v*dt
        self.delta_u[stuck==1]=0 #Arrest sets delta_u to zero
        return
    