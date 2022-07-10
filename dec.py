import numpy as np
from PIL import Image
import ncnn
net = ncnn.Net()
net.opt.use_vulkan_compute = True
net.load_param("/tmp/vq.param")
net.load_model("/tmp/vq.bin")

dumped_seqs=np.fromfile('oz.bin',dtype=np.uint16).astype(np.int32).reshape((-1,256))

k=0
for seq in dumped_seqs:
  with net.create_extractor() as ex:
    ex.input("in0", ncnn.Mat(seq).clone())
    hrr, out0 = ex.extract("out0")
  Image.fromarray(np.array(out0).astype(np.uint8)).save('%d.png'%k)
  k+=1
