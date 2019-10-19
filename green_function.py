# coding: utf-8
# (c) 2019-01-21 Teruhisa Okada
# (c) 2019-10-03 Teruhisa Okada update

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    

class GreenFunction():
    
    def __init__(self, param0, state0, obs, obs_error, back_error=None):
        assert len(state0) == len(obs)
        
        self.param_b = np.asarray(param0)
        self.state_b = np.asarray(state0)
        self.obs = obs
        self.Nparam = len(param0)
        self.Nstate = len(state0)
        self.Nobs = len(obs)
        self.params = []
        self.states = []
        self.d_param = np.zeros_like(param0)
        
        self.Rinv = np.eye(self.Nobs) / (obs_error**2)
        
        if back_error is not None:
            self.back_error = np.asarray(back_error)
            
    def append(self, param, state):
        self.params.append(param)
        self.states.append(state)
        
    def update(self, param, state):
        self.params.append(self.param_b)
        self.states.append(self.state_b)
        self.param_b = param
        self.state_b = state
        
    def reset(self, param, state):
        self.param_b = np.asarray(param)
        self.state_b = np.asarray(state)
        self.params = []
        self.states = []
        
    def matmul(self, a, b, c=None):
        if c is None:
            return np.matmul(a, b)
        else:
            return np.matmul(np.matmul(a, b), c)
        
    def get_ini_params(self):
        param = []
        for i in range(self.Nparam):
            p = self.param_b + np.eye(self.Nparam)[i] * self.back_error
            param.append(p)
        return param
        
    def solve_ls(self):
        #J = 1/2 (M(a)-y) R^-1 (M(a)-y)
        #dJ/da = G R^-1 (M(a0)+Gda-y) = 0
        #da = -(G R^-1 G)^-1 [G R^-1 (M(a0)-y)]
            
        s = [np.asarray(p) - self.param_b for p in self.params]
        S = np.asarray(s)
        NS = len(S)
        
        #T=S'(SS')^-1
        SSinv = np.linalg.inv(np.matmul(S.T, S))
        T = np.matmul(S, SSinv)
        
        #gj=sumi[(G(a1+si)-G(a1))*Tij]
        g = [np.sum(((self.states[i] - self.state_b) * T[i,j]) for i in range(NS)) 
             for j in range(self.Nparam)]
        G = np.asarray(g)
        
        #(G'R^-1G)^-1
        GRG = self.matmul(G, self.Rinv, G.T)
        l = np.linalg.inv(GRG)
        l2 = (l + l.T) / 2.0
        
        #G'R^-1 (Ga1-y)
        r = self.matmul(G, self.Rinv, (self.state_b-self.obs))
        
        #da=-(G'R^-1G)^-1 [G'R^-1 (Ga1-y)]
        self.d_param = - np.matmul(l2, r)
        param = self.param_b + self.d_param
        
        return param
        
    def solve_ridge(self):
        #J = 1/2 (a-a0)'B^-1(a-a0) + 1/2 (M(a)-y)'R^-1(M(a)-y)
        #dJ/dda1 = B^-1 da1 + G'R^-1 (M(a0)+Gda1-y)
        #da1 = -(B^-1 + G'R^-1G)^-1 [G'R^-1(M(a0)-y)]
        
        #dJ/da2 = B^-1 (da1+da2) + G'R^-1 (M(a1)+Gda2-y)
        #da2 = -(B^-1 + G'R^-1G)^-1 [B^-1 da1 + G'R^-1(M(a1)-y)]
            
        s = [np.asarray(p) - self.param_b for p in self.params]
        S = np.asarray(s)
        NS = len(S)
        
        # T = S'(SS')^-1
        SSinv = np.linalg.inv(np.matmul(S.T, S))
        T = np.matmul(S, SSinv)
        
        # gj=sumi[(M(a1+si)-M(a1))*Tij]
        g = [np.sum(((self.states[i] - self.state_b) * T[i,j]) for i in range(NS)) 
             for j in range(self.Nparam)]
        G = np.asarray(g)
        
        # L = (B^-1 + G'R^-1G)^-1
        GRG = self.matmul(G, self.Rinv, G.T)
        Binv = np.eye(self.Nparam) / (self.back_error**2)
        l = np.linalg.inv(Binv + GRG)
        l2 = (l + l.T) / 2.0
        
        # R = B^-1 da1 + G'R^-1 (M(a1)-y)
        r = self.matmul(G, self.Rinv, (self.state_b-self.obs))
        r += np.matmul(Binv, self.d_param)
        
        # da2 = -L R
        self.d_param = - np.matmul(l2, r)
        param = self.param_b + self.d_param
        
        return param
        
    def cost(self, state): 
        # (Ga-y)' R^-1 (Ga-y)
        misfit = state - self.obs
        return np.matmul(np.matmul(misfit, self.Rinv), misfit)


class GreenFunctionSparse(GreenFunction):
        
    def solve_sparse_ADMM(self, Nloop=20, lam=1., mu=1.):
        #J = 1/2 (M(a)-y)'R^-1(M(a)-y) + λ|a|
        #J = 1/2λ (M(af)-y)'R^-1(M(af)-y) + |ag| + mu/2 (af-agk+hk/mu)**2
        #dJ/ddaf = 1/λ G'R^-1(M(af0)+Gdaf-y) + mu(af0+daf-agk+hk/mu)
        #daf = -(1/λ G'R^-1G + muI)^-1 [1/λ G'R^-1(M(af0)-y) + mu(af0-agk+hk/mu)]
        
        param_f = self.param_b.copy()
        param_g = self.param_b.copy()
        u = np.zeros(self.Nparam)
        
        for i in range(Nloop):
            param_f = self._solve_f(self.param_b, self.state_b, param_g, u, lam, mu)
            param_g = self._soft_threshold(param_f + u, 1./mu)
            u = u + param_f - param_g
            
        return param_g
                    
    def _solve_f(self, param_b, state_b, param_g, u, lam, mu):
        #daf = -(1/λ G'R^-1G + muI)^-1 [1/λ G'R^-1(M(af0)-y) + mu(af0-agk+hk/mu)]
        
        s = [np.asarray(p) - param_b for p in self.params]
        S = np.asarray(s)
        NS = len(S)

        # T = S'(SS')^-1
        SSinv = np.linalg.inv(np.matmul(S.T, S))
        T = np.matmul(S, SSinv)

        # gj = sumi[(G(a1+si)-G(a1))*Tij]
        g = [np.sum(((self.states[i] - state_b) * T[i,j]) for i in range(NS)) 
             for j in range(self.Nparam)]
        G = np.asarray(g)

        # L = (1/λ G'R^-1G + muI)^-1
        GRG = self.matmul(G, self.Rinv, G.T)
        l = np.linalg.inv(1/lam * GRG + mu * np.eye(self.Nparam))
        l2 = (l + l.T) / 2.0

        # R = 1/λ G'R^-1(M(af0)-y) + mu(af0-agk+hk/mu)
        r = self.matmul(G, self.Rinv, (state_b-self.obs))
        r += mu * (param_b - param_g + u)

        # daf = -L R
        param = param_b - np.matmul(l2, r)

        return param
    
    def _soft_threshold(self, x, c):
        y = np.zeros(len(x))
        y = np.where(x>c, x-c, y)
        y = np.where(x<-c, x+c, y)
        return y
        
    def solve_sparse(self, lam=1.0, mu=1.0):
        #J = 1/2 (M(a)-y)'R^-1(M(a)-y) + λ|a|
        #dJ/dda = 1/λ G'R^-1(M(ab)+Gda-y) + sign(ab+da)
        #da1 = -(1/λ G'R^-1G + sign(ab+da))^-1 [1/λ G'R^-1(M(ab)-y)]
            
        s = [np.asarray(p) - self.param_b for p in self.params]
        S = np.asarray(s)
        NS = len(S)
        
        #T=S'(SS')^-1
        SSinv = np.linalg.inv(np.matmul(S.T, S))
        T = np.matmul(S, SSinv)
        
        #gj=sumi[(G(a1+si)-G(a1))*Tij]
        g = [np.sum(((self.states[i] - self.state_b) * T[i,j]) for i in range(NS)) 
             for j in range(self.Nparam)]
        G = np.asarray(g)
        
        ##(B^-1 + G'R^-1G)^-1
        GRG = self.matmul(G, self.Rinv, G.T)
        l = np.linalg.inv(1/lam * GRG + np.sign(self.param_b))
        l2 = (l + l.T) / 2.0
        
        ##B^-1 da1 + G'R^-1 (Ga1-y)
        r = 1/lam * self.matmul(G, self.Rinv, (self.state_b-self.obs))
        
        #da=-(B^-1 + G'R^-1G)^-1 [G'R^-1 (Ga1-y)]
        self.d_param = -np.matmul(l2, r)
        param = self.param_b + self.d_param 
        
        return param

    def solve_ElasticNet(self):
        
        """
        J = 1/2 (M(a)-y)'R^-1(M(a)-y) + λΣ{α|a|+(1-α)a^2}
        J = 1/2 (M(a)-y)'R^-1(M(a)-y) + 1/2 (a-a0)'B^-1(a-a0) + λ|a|
        
        dJ/dda = 1/λ G'R^-1(M(ab)+Gda-y) + sign(ab+da)
        da1 = -(1/λ G'R^-1G + sign(ab+da))^-1 [1/λ G'R^-1(M(ab)-y)]
        """
        