from lib.include import *
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


#----------

# from https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
# https://gist.github.com/post2web/a92be14008646a3d10b4183c8d35375f
def as_stride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0,s1=arr.strides[:2]
    m1,n1=arr.shape[:2]
    m2,n2=sub_shape
    view_shape=(1+(m1-m2)//stride[0],1+(n1-n2)//stride[1],m2,n2)+arr.shape[2:]
    strides=(stride[0]*s0,stride[1]*s1,s0,s1)+arr.strides[2:]
    subs=np.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
    return subs



def np_pooling (data, kernel_size, stride=None, padding=False, method='max'):

    '''Overlapping pooling on 2D or 3D data.
    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).
    Return <result>: pooled matrix.
    '''

    m, n  = data.shape[:2]
    ky,kx = kernel_size
    if stride is None:
        stride=(ky,kx)
    sy,sx=stride

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if padding:
        ny=_ceil(m,sy)
        nx=_ceil(n,sx)
        size=((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=data
    else:
        mat_pad=data[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]

    view = as_stride(mat_pad,kernel_size,stride)

    if method=='max':
        pool = np.nanmax(view,axis=(2,3))

    elif method=='mean':
        pool = np.nanmean(view,axis=(2,3))

    else:
        raise NotImplementedError

    return pool

#----------

def np_loss_cross_entropy(probability, truth):
    batch_size = len(probability)
    truth = truth.reshape(-1)
    p = probability[np.arange(batch_size),truth]
    loss = -np.log(np.clip(p,1e-6,1))
    loss = loss.mean()
    return loss


def np_loss_binary_cross_entropy(probability, truth):
    batch_size = len(probability)
    probability = probability.reshape(-1)
    truth = truth.reshape(-1)

    log_p_pos = -np.log(np.clip(probability,1e-5,1))
    log_p_neg = -np.log(np.clip(1-probability,1e-5,1))

    loss = log_p_pos[truth==1].sum() + log_p_neg[truth==0].sum()
    loss = loss/len(truth)
    return loss

def np_onehot(x, num_class):
    x = x.reshape(-1).astype(np.int32)
    onehot = np.zeros((len(x), num_class))
    onehot[np.arange(len(x)), x] = 1
    return onehot



def np_metric_accuracy(predict, truth):
    truth = truth.reshape(-1)
    predict = predict.reshape(-1)
    correct = truth==predict
    correct = correct.mean()
    return correct

def np_metric_roc_auc(probability, truth):
    truth = truth.reshape(-1)
    probability = probability.reshape(-1)
    score = roc_auc_score(truth, probability)
    return score


def np_metric_tp_fp(probability, truth, threshold=0.5):
    truth = truth.reshape(-1)
    probability = probability.reshape(-1)
    predict = (probability>threshold).astype(np.float32)

    num_pos = truth.sum()
    num_neg = len(truth) - num_pos
    tpr = ((truth==1)*(predict==1)).sum()/num_pos
    fpr = ((truth==0)*(predict==1)).sum()/num_neg
    return tpr,fpr



def np_metric_top_k(probability, truth, k=2):
    batch_size,num_class = probability.shape
    truth = truth.reshape(-1,1)
    predict = np.argsort(-probability,-1)
    correct = truth==predict
    correct = correct.mean(0)
    correct = correct.cumsum()
    return correct[:k]

# https://stackoverflow.com/questions/28339746/equal-error-rate-in-python

def np_metric_eer(probability, truth):
    fpr, tpr, threshold = roc_curve(truth, probability)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    t = interp1d(fpr, threshold)(eer)
    return eer, t