import sys
sys.path.append("/home/kaihsinwu/CYTNX075/")

import cytnx as cy
import numpy as np

# prepare MPS:
def productMPS(init_s):
    ## init_s: [0,1,0,...] binary value of initial state
    ## each site bond order, [virtL,virtR,phys]
    
    out = []
    for i in range(len(init_s)):
        tmp = cy.zeros(2)
        tmp[init_s[i]] = 1
        tmp.reshape_(1,2,1)
        #print(tmp) 
        out.append(cy.UniTensor(tmp,1))
        out[-1].set_labels([-i-1,i,-i-2])
        

    return out



def print_mps(mps):
    for i in mps:
        i.print_diagram()
        print(i)

    return 0


def norm2(mps):
    # calculate <psi|psi>

    Lmin = mps[-1].labels()[2]-1
    #print(Lmin)
    A = mps[0]
    At = mps[0].relabel(2,Lmin)
    R = cy.Contract(A,At)

    for i in range(1,len(mps)):
        A = mps[i]
        At = mps[i].relabel(0,Lmin)
        At = At.relabel(2,Lmin-1)
        Lmin-=1
        R = cy.Contract(cy.Contract(R,A),At)

    #R.print_diagram()
    return R.Trace().item()

## make it left normalized form:
def make_Lform(state):
    
    R = None
    tmp = None
    for i in range(len(state)):
        if R is None:
            tmp = state[i]
        else:
            tmp = cy.Contract(R,state[i])

        #tmp.print_diagram()
        tmp.set_rowrank(2)
        out = cy.linalg.Svd(tmp)
         
        out[2] = cy.Contract(out[0],out[2])
        out[1].set_rowrank(1)
        
        #out[1].print_diagram()
        #out[2].print_diagram()

        #out[1].set_label(2,out[1].labels()[0]-1)        
        #out[2].set_label(0,out[1].labels()[2])

        state[i] = out[1]
        R = out[2] 
        R.set_label(0,R.labels()[0]-1)
        state[i].set_labels([-i-1,i,-i-2])


    

    return state

def make_Rform(state):

    L = None
    tmp = None
    for i in range(len(state)):
        if L is None:
            tmp = state[-i-1]
        else:
            tmp = cy.Contract(state[-i-1],L)

        #tmp.print_diagram()
        tmp.set_rowrank(1)
        out = cy.linalg.Svd(tmp)
         
        out[2].set_rowrank(1)
        out[1] = cy.Contract(out[1],out[0])
        
        state[-i-1] = out[2]
        L = out[1] 
        #L.set_label(1,L.labels()[0]-5)
        state[-i-1].set_labels([-len(state)+i,len(state)-i-1,-len(state)+i-1])


    

    return state


def get_svals(mps):
    # this will make the mps into it's Lform

    state = [s.clone() for s in mps]

    state = make_Rform(state)    

    svals = []
    R = None
    tmp = None
    for i in range(len(state)):
        if R is None:
            tmp = state[i]
        else:
            tmp = cy.Contract(R,state[i])

        #tmp.print_diagram()
        tmp.set_rowrank(2)
        out = cy.linalg.Svd(tmp)

        svals.append(out[0].get_block_())
         
        out[2] = cy.Contract(out[0],out[2])
        out[1].set_rowrank(1)
        


        state[i] = out[1]
        R = out[2] 
        R.set_label(0,R.labels()[0]-5)
        state[i].set_labels([-i-1,i,-i-2])

    return svals

def entanglement(svals):
    Sout = []
    for i in range(len(svals)):
        tmp = 0
        #print(i,svals[i])
        for n in range(svals[i].shape()[0]):
            lbd2 = np.abs(svals[i][n].item())**2
            if lbd2 < 1.0e-12:
                tmp += 0 
            else:
                tmp += -lbd2*np.log(lbd2)

        Sout.append(tmp)
        
    return Sout

def get_entanglement(mps):

    svals = get_svals(mps)
    #print(svals)
    Sout = entanglement(svals)

    return Sout


def get_amp(mps,targ):

    out = None
    for i in range(len(mps)):
        #print(i)
        #state[i].print_diagram()
        A = mps[i].get_block_()[:,targ[i],:]

        if i==0:
            A.reshape_(1,A.shape()[0])

        #if i==2:
        #    print(out)
        #    print(A)
 
        if out is None:
            out = A
        else:
            out = cy.linalg.Dot(out.clone(),A)        

        #if i==2:
        #    print(out)
        #print(i,A)
        #print("out",out)

    return out.item()

def multi_Op(mps,mask,Op):
    # ipt is the input, MPS
    # mask: list, if element=1, no operate of H

    for i in range(len(mask)):
        if mask[i]==0:
            Op.set_labels([mps[i].labels()[1],mps[i].labels()[1]+1])
            mps[i] = cy.Contract(mps[i],Op)
            mps[i].set_label(2,Op.labels()[0]) 
            mps[i].permute_([0,2,1])           
            
    return mps



