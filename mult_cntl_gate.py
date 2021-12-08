#import cytnx as cy
from tools import *
import numpy as np 
## study the decomp from NC 4.10 

def multi_cntl_gate(state,gate,Xmax=None,trunc_err=0,is_measure=False,verbose=False): 
    # assuming gate operate on the last site. 
    # assuming state is in NN friendly form.
    # assuming state tis in Lform. (v3)    
    
    # Xmax: maximum of vbond dimension 
    # Trunc_err: setting the truncation err

    is_trunc = False
    max_vbdim = 0


    if Xmax is not None:
        is_trunc = True
    else:
        Xmax = 9999999 ## some large value

    if trunc_err > 0:
        is_trunc = True

    
    n = int(len(state)/2+1)

    Tf = cy.qgates.toffoli();

    Sout=[]
    for i in range(n-2):

        A=state[i*2]
        B=state[i*2+1]
        C=state[i*2+2]
        phys_lbl = [A.labels()[1],B.labels()[1],C.labels()[1]]
        phys_lbl_out = [A.labels()[1]+3,B.labels()[1]+3,C.labels()[1]+3]
        
        Tf.set_labels(phys_lbl+phys_lbl_out)
      
        tmp = cy.Contract(cy.Contract(cy.Contract(A,B),C),Tf)
        
        new_lbl = [tmp.labels()[0],tmp.labels()[1]] + phys_lbl

        tmp.set_labels(new_lbl)
        tmp.permute_([0,2,3,4,1],rowrank=2)
        #tmp.contiguous_()
             
        ## go back to PS form, or truncate:
        if is_trunc:
            out = cy.linalg.Svd_truncate(tmp,keepdim=Xmax,err=trunc_err)
            
        else:
            out = cy.linalg.Svd(tmp)

        if out[0].shape()[0] > max_vbdim:
            max_vbdim = out[0].shape()[0]
       
 
        A = out[1]
        tmp = cy.Contract(out[0],out[2])

        tmp.set_rowrank(2)
        if is_trunc:
            out = cy.linalg.Svd_truncate(tmp,keepdim=Xmax,err=trunc_err)
        else:
            out = cy.linalg.Svd(tmp)

        if out[0].shape()[0] > max_vbdim:
            max_vbdim = out[0].shape()[0]

        #print(out[0])
        B = out[1]
        C = cy.Contract(out[0],out[2])

        
        #A.print_diagram()
        #B.print_diagram()
        #C.print_diagram()

        ## relabel:
        A.set_label(2,A.labels()[0]-1)
        B.set_label(0,A.labels()[2])
        B.set_label(2,B.labels()[0]-1)
        C.set_label(0,B.labels()[2])
        
        A.set_rowrank(1)
        B.set_rowrank(1)
        C.set_rowrank(1)


        #R = cy.Contract(cy.Contract(A,B),C)

        state[i*2] = A
        state[i*2+1]=B
        state[i*2+2]=C
        
        if is_measure:
            Sv = get_entanglement(state)
            Sout.append(Sv)
    
    ## cntl gate apply:
    CG = cy.qgates.cntl_gate_2q(gate)
    
    #print("CG",CG.get_block_().reshape(4,4))
    phys_lbl = [state[-2].labels()[1],state[-1].labels()[1]]
    phys_lbl_out = [state[-2].labels()[1]+2,state[-1].labels()[1]+2]
    CG.set_labels(phys_lbl+phys_lbl_out)
    tmp = cy.Contract(cy.Contract(state[-2],state[-1]),CG)

    new_lbl = [tmp.labels()[0],tmp.labels()[1]] + phys_lbl
    tmp.set_labels(new_lbl)
    tmp.permute_([0,2,3,1],rowrank=2)

    if is_trunc:
        out = cy.linalg.Svd_truncate(tmp,keepdim=Xmax,err=trunc_err)
    else:
        out = cy.linalg.Svd(tmp) 

    if out[0].shape()[0] > max_vbdim:
        max_vbdim = out[0].shape()[0]

    state[-2] = out[1].relabel(2,out[1].labels()[0]-1)
    state[-1] = cy.Contract(out[0],out[2])
    state[-1].set_label(0,state[-2].labels()[2])
        

    state[-2].set_rowrank(1)
    state[-1].set_rowrank(1)
   
    if is_measure:
        Sv = get_entanglement(state)
        Sout.append(Sv)
 
    ## ladder up:
    for i in range(n-2):

        C=state[-i*2-2]
        B=state[-i*2-3]
        A=state[-i*2-4]
        phys_lbl = [A.labels()[1],B.labels()[1],C.labels()[1]]
        phys_lbl_out = [A.labels()[1]+3,B.labels()[1]+3,C.labels()[1]+3]
        
        Tf.set_labels(phys_lbl+phys_lbl_out)
        tmp = cy.Contract(cy.Contract(cy.Contract(A,B),C),Tf)

        
        new_lbl = [tmp.labels()[0],tmp.labels()[1]] + phys_lbl

        tmp.set_labels(new_lbl)
        tmp.permute_([0,2,3,4,1],rowrank=3)

              
        ## go back to PS form, or truncate:
        if is_trunc:
            out = cy.linalg.Svd_truncate(tmp,keepdim=Xmax,err=trunc_err)
        else:
            out = cy.linalg.Svd(tmp)
        if out[0].shape()[0] > max_vbdim:
            max_vbdim = out[0].shape()[0]
        C = out[2]
        #C.print_diagram()
        tmp = cy.Contract(out[1],out[0])

        tmp.set_rowrank(2)
        if is_trunc:
            out = cy.linalg.Svd_truncate(tmp,keepdim=Xmax,err=trunc_err)
        else:
            out = cy.linalg.Svd(tmp)
        if out[0].shape()[0] > max_vbdim:
            max_vbdim = out[0].shape()[0]
        B = out[2]
        A = cy.Contract(out[1],out[0])


        ## relabel:
        A.set_label(2,A.labels()[0]-1)
        B.set_label(0,A.labels()[2])
        B.set_label(2,B.labels()[0]-1)
        C.set_label(0,B.labels()[2])

        A.set_rowrank(1)
        B.set_rowrank(1)
        C.set_rowrank(1)

        state[-i*2-2]=C

        state[-i*2-3]=B
        state[-i*2-4]=A

        if is_measure:
            Sv = get_entanglement(state)
            Sout.append(Sv)

    if verbose:
        print("[truncate? %d][multi-cntl] Maxdim = %d "%(is_trunc,max_vbdim))
    
    if is_measure:
        return state,Sout
    else:
        return state


if __name__=="__main__":

    Nc = 3
    Nv = Nc-2
    N = Nc + Nv

    H = cy.qgates.hadamard().get_block()
    H = cy.UniTensor(H.real(),1)/np.sqrt(2)

    X = cy.UniTensor(cy.physics.pauli('x').real(),1)
    Z = cy.UniTensor(cy.physics.pauli('z').real(),1)
    I = cy.UniTensor(cy.eye(2),1)
    


    # * arrangement:  c_i:claus bits, v_i: ancilla bits
    #     c 1.c2.v1.c3.v2.c4.v3.....v_{n-1}.c_n
    state = productMPS([0,0,0,0])

    PVF_measure_amp(state)
    
    state = multi_Op(state,[0,0,1,0],H)
    exit(1)
    PVF_measure_amp(state)

    state = multi_cntl_gate(state,Z)
    
    #print_mps(state)
    state2 = make_Lform(state)
    
    #print_mps(state2)
    #R = cy.Contract(cy.Contract(cy.Contract(state[0],state[1]),state[2]),state[3])

    #print(R.get_block_().reshape(16))

    #print(get_amp(state,[1,1,1,1]))
    #PVF_measure_amp(state)



