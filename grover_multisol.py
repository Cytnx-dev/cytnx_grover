from tools import *
from mult_cntl_gate import *
import numpy as np

H = cy.qgates.hadamard().get_block()
H = cy.UniTensor(H.real(),1)/np.sqrt(2)
X = cy.UniTensor(cy.physics.pauli('x').real(),1)
Z = cy.UniTensor(cy.physics.pauli('z').real(),1)
I = cy.UniTensor(cy.eye(2),1)


def Oracle_singl_gate(ipt,mark_state,Xmax=None,trunc_err=0,verbose=False):
    if verbose:
        print("[oracle]",end="")

    state = multi_Op(ipt,mark_state,X)
    ## multi-cntl-Z
    state = multi_cntl_gate(state,Z,Xmax=Xmax,trunc_err=trunc_err,verbose=verbose)
    
    state = multi_Op(state,mark_state,X)

    return state

def Oracle_multi_gate(ipt,N_mark_state,Xmax=None,trunc_err=0,verbose=False):
    if verbose:
        print("[multi-oracle][r=%d]"%(N_mark_state),end="")

    # checking:
    n = int(len(ipt)/2+1)

    nRC = None
    for i in range(n):
        if N_mark_state == 2**i:
            nRC = i
            break;
    
    if nRC is None:
        raise ValueError("N_mark_state must be power of 2.")


    #state = multi_Op(ipt,mark_state,X)

    ## multi-cntl-Z
    pstate = [ipt[i] for i in range(len(ipt)-2*nRC)]

    pstate = multi_cntl_gate(pstate,Z,Xmax=Xmax,trunc_err=trunc_err,verbose=verbose)
    
    for i in range(len(ipt)-2*nRC):
        ipt[i] = pstate[i]

    #state = multi_Op(state,mark_state,X)

    return ipt
    

def Amplifier(ipt,amask,Xmax=None,trunc_err=0,verbose=False):
    if verbose:
        print("[amplifier]",end="")
    
    #nullmask = [0]*len(ipt)
    state = multi_Op(ipt,amask,H)
    state = multi_Op(state,amask,X)

    ## multi-cntl-Z
    state = multi_cntl_gate(state,Z,Xmax=Xmax,trunc_err=trunc_err,verbose=verbose)

    state = multi_Op(state,amask,X)
    state = multi_Op(state,amask,H)

    return state


def create_target_mask(targ,n_tot):

    tmask = np.zeros(n_tot,dtype=np.int)
    tmask[0] = targ[0]
    tmp=1
    for i in range(1,n_tot,2):
        #print(i,tmp)
        tmask[i] = targ[tmp]
        tmp+=1

    return tmask

def Full_measure_amp(mps,n):

    n_tot = len(mps)
    pall=[]    
    for i in range(2**n):
        cstate=[int(x) for x in list(format(i,'0%db'%(n)))]
        #print(cstate)
        tmask=create_target_mask(cstate,n_tot)
        p = get_amp(mps,tmask)
        #print(i,p)
        pall.append(p)
    return pall

def PVF_measure_amp(mps):
    n = len(mps)
    pall=[]    
    for i in range(2**n):
        cstate=[int(x) for x in list(format(i,'0%db'%(n)))]
        print(cstate)
        #tmask=create_target_mask(cstate,n_tot)
        p = get_amp(mps,cstate)
        print(i,p)
        pall.append(p)
    return pall

class Grover:
    def __init__(self,nbits,target):
        # 0. prepare bits mask:
        self.n = nbits # claus qubits
        self.targ = target         
        self.n_tot = 2*self.n-2 # total qubits 

        ## create ancilla bits mask:
        self.amask = np.zeros(self.n_tot)
        self.amask[2::2] = 1
        
        ## create target mask:
        self.tmask = create_target_mask(self.targ,self.n_tot)
        
        # 1. create init mps 
        self.mps = productMPS([0]*self.n_tot) 
        
        #2. into superpose:
        self.mps = multi_Op(self.mps,self.amask,H)
           
    def apply_oracle(self,Xmax=None,trunc_err=0,verbose=False):
        self.mps = Oracle_singl_gate(self.mps,self.tmask+self.amask,Xmax=Xmax,trunc_err=trunc_err,verbose=verbose)

        return self        

    def apply_amplifier(self,Xmax=None,trunc_err=0,verbose=False):
        self.mps = Amplifier(self.mps,self.amask,Xmax=Xmax,trunc_err=trunc_err,verbose=verbose)
        
        return self

    def step(self,Xmax=None,trunc_err=0,verbose=False): 
        # operate a full step
            
        self.apply_oracle(Xmax,trunc_err,verbose)
        self.apply_amplifier(Xmax_trunc_err,verbose)

        return self


    def measure_pamp(self, cstate, is_raw=False):
        # cstate is classical state: list of nbits
        # if is_raw = True, cstate should contain 2*nbits-2 elements (includeing ancilla)
        #    is_raw = False (default), cstate should have nbits elements (only claus bits)
        pa = 0.
        if is_raw:
            if len(cstate) != self.n_tot:
                raise ValueError("is_raw=True should have len(cstate)=2n-2")

            pa = get_amp(self.mps,cstate)
                
        else:
            if len(cstate) != self.n:
                raise ValueError("is_raw=False should have len(cstate)=n")
            msk=create_target_mask(cstate,self.n_tot)
            pa = get_amp(self.mps,msk)

        return pa

class Grover_multi:
    def __init__(self,nbits,Ntarget):
        # 0. prepare bits mask:
        self.n = nbits # claus qubits
        self.Ntarget = Ntarget        
        self.n_tot = 2*self.n-2 # total qubits 

        ## create ancilla bits mask:
        self.amask = np.zeros(self.n_tot)
        self.amask[2::2] = 1
        
        ## create target mask:
        #self.tmask = create_target_mask(self.targ,self.n_tot)
        
        # 1. create init mps 
        self.mps = productMPS([0]*self.n_tot) 
        
        #2. into superpose:
        self.mps = multi_Op(self.mps,self.amask,H)
           
    def apply_oracle(self,Xmax=None,trunc_err=0,verbose=False):
        self.mps = Oracle_multi_gate(self.mps,self.Ntarget,Xmax=Xmax,trunc_err=trunc_err,verbose=verbose)

        return self        

    def apply_amplifier(self,Xmax=None,trunc_err=0,verbose=False):
        self.mps = Amplifier(self.mps,self.amask,Xmax=Xmax,trunc_err=trunc_err,verbose=verbose)
        
        return self

    def step(self,Xmax=None,trunc_err=0,verbose=False): 
        # operate a full step
            
        self.apply_oracle(Xmax,trunc_err,verbose)
        self.apply_amplifier(Xmax_trunc_err,verbose)

        return self


    def measure_pamp(self, cstate, is_raw=False):
        # cstate is classical state: list of nbits
        # if is_raw = True, cstate should contain 2*nbits-2 elements (includeing ancilla)
        #    is_raw = False (default), cstate should have nbits elements (only claus bits)
        pa = 0.
        if is_raw:
            if len(cstate) != self.n_tot:
                raise ValueError("is_raw=True should have len(cstate)=2n-2")

            pa = get_amp(self.mps,cstate)
                
        else:
            if len(cstate) != self.n:
                raise ValueError("is_raw=False should have len(cstate)=n")
            msk=create_target_mask(cstate,self.n_tot)
            pa = get_amp(self.mps,msk)

        return pa




n = 8
targ = [1,1,1,1,1,1,1,1]
Ntarget = 4
Nstep = 40
Thres=1.0e-12


prob = []
prob_all = []
Sall = []
circ = Grover_multi(n,Ntarget)

#measure:
Sv = get_entanglement(circ.mps)
Sall.append(Sv)
prob.append(circ.measure_pamp(targ)**2)
prob_all.append(Full_measure_amp(circ.mps,n))


for i in range(Nstep):
    #circ.step()
    circ.apply_oracle(trunc_err=Thres,verbose=True)

    #exit(1)
    #measure:
    Sv = get_entanglement(circ.mps)
    Sall.append(Sv)
    
    
    circ.apply_amplifier(trunc_err=Thres,verbose=True)
    
    #measure:
    Sv = get_entanglement(circ.mps)
    Sall.append(Sv)

    #meausre: 
    prob.append(circ.measure_pamp(targ)**2)
    prob_all.append(Full_measure_amp(circ.mps,n))
    print(i+1,":",prob[-1])
    #print(Sv)


prob_all = np.array(prob_all)**2
Sall = np.array(Sall)[:,:-1]

np.save("n%dr%d_ts"%(n,Ntarget),prob)
np.save("n%dr%d_ts_all"%(n,Ntarget),prob_all)
np.save("n%dr%d_ts_Sall"%(n,Ntarget),Sall)
