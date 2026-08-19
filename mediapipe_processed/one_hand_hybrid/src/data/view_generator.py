from __future__ import annotations
import torch

RIGHT=slice(50,92); LEFT=slice(92,134)
MODES={"full":(1,1),"right_only":(1,0),"left_only":(0,1)}

def make_view(features, detected_mask, mode, dropout=None):
    """Return masked features, view mask and valid mask in [right,left] order."""
    if mode not in MODES: raise ValueError(f"unknown view mode: {mode}")
    x=features.clone(); r,l=MODES[mode]
    shape=(*detected_mask.shape[:-1],2)
    view=torch.empty(shape,dtype=detected_mask.dtype,device=detected_mask.device)
    view[...,0]=r; view[...,1]=l
    if not r: x[...,RIGHT]=0
    if not l: x[...,LEFT]=0
    valid=detected_mask.to(view.dtype)*view
    if dropout is not None:
        valid=valid*dropout.to(valid.dtype)
        x[...,RIGHT]=x[...,RIGHT]*valid[...,0,None]
        x[...,LEFT]=x[...,LEFT]*valid[...,1,None]
    return x,view,valid
