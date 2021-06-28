from lib.include import *

# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
def soft_dice_criterion(logit, truth):
    batch_size = len(logit)
    probability = torch.sigmoid(logit)

    p = probability.view(batch_size,-1)
    t = truth.view(batch_size,-1)

    p = p*2-1
    t = t*2-1

    #non-empty
    intersection = (p * t).sum(-1)
    union =  (p * p).sum(-1) + (t * t).sum(-1)
    dice  = 1 - 2*intersection/union

    loss = dice
    return loss

def soft_dice1_criterion(logit, truth):
    batch_size = len(logit)
    probability = torch.sigmoid(logit)

    p = probability.view(batch_size,-1)
    t = truth.view(batch_size,-1)

    p = p*2-1
    t = t*2-1

    #non-empty
    intersection = (p * t).sum(-1)
    union =  (p * p).sum(-1) + (t * t).sum(-1)
    dice  =  2*intersection/union

    eps  = 1e-12
    dice = torch.clamp(dice,eps,1-eps)


    loss = -torch.log(dice)
    return loss
