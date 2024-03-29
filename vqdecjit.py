import torch
from PIL import Image
import numpy as np

def initvqdecjit():
  global vqdec
  global vqd_shape
  vqdec=torch.jit.load('/content/vqdec.pt')
  vqd_shape = torch.tensor([16,-1],dtype=torch.long)
  

def showp(n, prt=False):
  global mlat
  global mfn
  daaz=dumped_seqs[n]
  #out0 = vqdec(torch.flatten(torch.from_numpy(daaz.astype(np.int32)).to(mindd.device) ),vqd_shape)
  out0 = vqdecjit.vqdec(torch.flatten(torch.from_numpy(daaz.astype(np.int32)).to(mindd.device) ),vqdecjit.vqd_shape)
  uz=Image.fromarray(out0.cpu().numpy().astype(np.uint8))
  uz.save('/content/sample_data/%d.png'%n)
  if prt:
    display(uz)
    mfn='%d'%n
    mlat=str(daaz)[1:-1].replace('\n','')
    return mlat
  else:
    return uz

def showp2(seq):
  out0 = vqdec(torch.flatten(torch.from_numpy(seq.astype(np.int32)).to(mindd.device)),vqd_shape)
  uz=Image.fromarray(out0.cpu().numpy().astype(np.uint8))
  uz.save('/content/sample_data/000.png')
  return uz


initvqdecjit()
