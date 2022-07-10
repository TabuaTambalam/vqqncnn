import numpy as np
from PIL import Image
import ncnn
net = ncnn.Net()
net.opt.use_vulkan_compute = True
net.load_param("/tmp/vq.param")
net.load_model("/tmp/vq.bin")

dumped_seqs=np.fromfile('oz.bin',dtype=np.uint16).astype(np.int32).reshape((-1,256))

def mkCBemb(seq):
  with net.create_extractor() as ex:
    ex.input("in0", ncnn.Mat(seq).clone())
    hrr, out0 = ex.extract("2")
  del ex
  return out0

def emb2img(emb):
  with net.create_extractor() as ex:
    ex.input("2", emb)
    hrr, out0 = ex.extract("out0")
  del ex
  return Image.fromarray(np.array(out0).astype(np.uint8))

def npmkCBemb(seq):
  with net.create_extractor() as ex:
    ex.input("in0", ncnn.Mat(seq).clone())
    hrr, out0 = ex.extract("2")
  del ex
  return np.array(out0)


def npemb2img(emb):
  with net.create_extractor() as ex:
    ex.input("2", ncnn.Mat(emb).clone())
    hrr, out0 = ex.extract("out0")
  del ex
  return Image.fromarray(np.array(out0).astype(np.uint8))

def interpo(seq1,seq2,step=30,scale=1.2,outfmt='%02d.png'):
  stp=step-1
  divi=stp/scale
  em1=npmkCBemb(seq1)
  em2=npmkCBemb(seq2)
  for i in range(step):
    npemb2img((em1*i+em2*(stp-i))/divi).save(outfmt%i)


interpo(dumped_seqs[300],dumped_seqs[200])
