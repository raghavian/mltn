import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mps import MPS
import pdb
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPS = 1e-6

class MLTN(nn.Module):
    def __init__(self, input_dim, output_dim, bond_dim, feature_dim=2, nCh=3,
                 kernel=2, virtual_dim=1,bn=False,dropout=0.0,seg=False,
                 adaptive_mode=False, periodic_bc=False, parallel_eval=False,
                 label_site=None, path=None, init_std=1e-9, use_bias=True,
                 fixed_bias=True, cutoff=1e-10, merge_threshold=2000):
        super().__init__()
        self.input_dim = input_dim
        nDim = len(self.input_dim)
        self.nDim = nDim    
        self.bn = bn
        self.dropout = dropout
        self.vdim = 1
        ## Work out the depth of the tensornet
        self.dropout = dropout  
        self.ker = [16,4]

        nL = np.log(self.input_dim[0].item())/np.log(self.ker[0])
        self.L = int(np.floor(nL))-1
        
        nCh =  self.ker[0]**nDim * nCh
        self.L = len(self.ker)
        print("Using depth of %d"%(self.L+1))
        self.nCh = nCh
        self.lFeat = 1
        feature_dim = self.lFeat*nCh 
        
        ### First level MPS blocks
        self.module = nn.ModuleList([MPS(input_dim=((self.vdim-1)*int(i>0)+1)*torch.prod(self.input_dim//(np.prod(self.ker[:i+1]))),
            output_dim=self.vdim*torch.prod(self.input_dim//np.prod(self.ker[:i+1])),
            bond_dim=bond_dim,lFeat=self.lFeat, 
            feature_dim=self.lFeat*(self.ker[i])**nDim, parallel_eval=parallel_eval,
            adaptive_mode=adaptive_mode, periodic_bc=periodic_bc) 
            for i in range(self.L)])
            
        if self.bn: 
            self.BN = nn.ModuleList([nn.BatchNorm1d(self.vdim*torch.prod(self.input_dim//(np.prod(self.ker[:i+1]))).numpy(),\
                                        affine=True) for i in range(self.L)])

        ### Third level MPS blocks
        ### Final MPS block
        self.mpsFinal = MPS(input_dim=self.vdim*torch.prod(self.input_dim//np.prod(self.ker)), 
                output_dim=output_dim,lFeat=self.lFeat,
                bond_dim=bond_dim, feature_dim=self.lFeat, 
                adaptive_mode=adaptive_mode, periodic_bc=periodic_bc, 
                parallel_eval=parallel_eval)

    def forward(self,x):

        b = [x.shape[0]] #Batch size

        for i in range(self.L):
                lKer = self.ker[i]
                iDim = self.input_dim // np.prod(self.ker[:i+1])
                if self.nDim == 2:
                    x = x.unfold(2,iDim[0],iDim[1]).unfold(3,iDim[0],iDim[1])
                else:
                    x = x.unfold(2,iDim[0],iDim[0]).unfold(3,iDim[1],iDim[1]).unfold(4,iDim[2],iDim[2])
            
                x = x.reshape(b[0],self.ker[i]**self.nDim,-1)
                if self.dropout > 0:
                    x = F.dropout(x,self.dropout,inplace=True)
                x = self.module[i](x)
                
                if self.bn and b[0] > 1:
                    x = self.BN[i](x)
                newRes = tuple(b) + tuple([self.vdim]) + tuple(iDim.numpy())
                x = x.view(newRes)

        # Final layer
        if self.dropout > 0:
            x = F.dropout(x,self.dropout,inplace=True)
        x = self.mpsFinal(x.view(b[0],1,-1))
        return x.view(b)


