# import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'

from lib.include import *
from lib.utility.draw import *
from lib.utility.file import *

from lib.include_torch import *
from lib.net.rate import *
from lib.net.layer_np import *



#---------------------------------------------------------------------------------
COMMON_STRING ='@%s:  \n' % os.path.basename(__file__)
if 1:
    seed = int(time.time())#335202   #5202  #123  #
    seed_py(seed)
    seed_torch(seed)

    torch.backends.cudnn.benchmark     = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.deterministic = True

    COMMON_STRING += '\tpytorch\n'
    COMMON_STRING += '\t\tseed = %d\n'%seed
    COMMON_STRING += '\t\ttorch.__version__              = %s\n'%torch.__version__
    COMMON_STRING += '\t\ttorch.version.cuda             = %s\n'%torch.version.cuda
    COMMON_STRING += '\t\ttorch.backends.cudnn.version() = %s\n'%torch.backends.cudnn.version()
    try:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = %s\n'%os.environ['CUDA_VISIBLE_DEVICES']
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        COMMON_STRING += '\t\tos[\'CUDA_VISIBLE_DEVICES\']     = None\n'
        NUM_CUDA_DEVICES = 1

    COMMON_STRING += '\t\ttorch.cuda.device_count()      = %d\n'%torch.cuda.device_count()
    COMMON_STRING += '\t\ttorch.cuda.get_device_properties() = %s\n' % str(torch.cuda.get_device_properties(0))[21:]

COMMON_STRING += '\n'

#---------------------------------------------------------------------------------
## useful : http://forums.fast.ai/t/model-visualization/12365/2

## pip install torch==1.4.0+cu101 torchvision==0.5.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
## pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp37-cp37m-linux_x86_64.whl
## pip install https://download.pytorch.org/whl/cu101/torchvision-0.5.0-cp37-cp37m-linux_x86_64.whl

if __name__ == '__main__':
    print (COMMON_STRING)
