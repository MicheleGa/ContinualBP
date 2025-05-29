import numpy as np
import random
from scipy.interpolate import CubicSpline
import torch


class Jitter(object):
    def __init__(self, prob=0.5, sigma=0.03):
        self.prob = prob
        self.sigma = sigma

    def __call__(self, inputs):
        if random.random() < self.prob:
            return inputs + torch.from_numpy(np.random.normal(loc=0., scale=self.sigma, size=inputs.shape)).float()
        else:
            return inputs
        
        
class Scaling(object):
    def __init__(self, prob=0.5, sigma=0.1):
        self.prob = prob
        self.sigma = sigma

    def __call__(self, inputs):
        if random.random() < self.prob:
            factor = np.random.normal(loc=1., scale=self.sigma, size=inputs.shape)
            res = np.multiply(inputs, factor)
            return res.float()
        else:
            return inputs
        
        
class Flip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs):
        if random.random() < self.prob:
            return -inputs.float()
        else:
            return inputs
        
    
class MagnitudeWarp(object):
    def __init__(self, prob=0.5, sigma=0.2, knot=4):
        self.prob = prob
        self.sigma = sigma
        self.knot = knot

    def __call__(self, inputs):
        if random.random() < self.prob:
            orig_steps = np.arange(inputs.shape[1])
            random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(inputs.shape[0], self.knot+2, inputs.shape[2]))
            warp_steps = (np.ones((inputs.shape[2],1))*(np.linspace(0, inputs.shape[1]-1., num=self.knot+2))).T
            ret = np.zeros_like(inputs)
            for i, pat in enumerate(inputs):
                warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(inputs.shape[2])]).T
                ret[i] = pat * warper
            
            return torch.from_numpy(ret).float()
        else:
            return inputs
        
    
class TimeWarp(object):
    def __init__(self, prob=0.5, sigma=0.2, knot=4):
        self.prob = prob
        self.sigma = sigma
        self.knot = knot

    def __call__(self, inputs):
        if random.random() < self.prob:
            orig_steps = np.arange(inputs.shape[1])
            random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(inputs.shape[0], self.knot+2, inputs.shape[2]))
            warp_steps = (np.ones((inputs.shape[2],1))*(np.linspace(0, inputs.shape[1]-1., num=self.knot+2))).T
            
            ret = np.zeros_like(inputs)
            for i, pat in enumerate(inputs):
                for dim in range(inputs.shape[2]):
                    time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
                    scale = (inputs.shape[1]-1)/time_warp[-1]
                    ret[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, inputs.shape[1]-1), pat[:,dim]).T
            return torch.from_numpy(ret.squeeze()).float()
        else:
            return inputs
        

class Identity(object):
    r"""Identity augmentation that keeps the input unchanged.
    """

    def __init__(self, prob=1.0):
        super(Identity, self).__init__()
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0, \
            "Probability must be a float in the range [0, 1]."
        self.prob = prob

    def __call__(self, inputs):
        """Applies the identity augmentation.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Unchanged input tensor.
        """
        return inputs
    

class RandomAugmentor(object):
    r"""Random data augmentations.

    This class applies a set of random augmentations to the input data based on
    their probabilities.

    
        augments (list of augment class):
            List of augmentation instances.

    Example:
        >>> augments_cfg = [
                Jitter(prob=0.2),
                Scaling(prob=0.2),
                Flip(prob=0.2),
                MagnitudeWarp(prob=0.2),
                TimeWarp(prob=0.2)
            ]
        >>> augments = RandomAugmentor(augments_cfg)
        >>> sig = torch.randn(256, 125, 1)
        >>> sig = augments(sig)
    """

    def __init__(self, augments):
        super(RandomAugmentor, self).__init__()

        self.augments = augments
        self.augment_probs = [aug.prob for aug in self.augments]

        has_identity = any([isinstance(aug, Identity) for aug in self.augments])
        if has_identity:
            assert sum(self.augment_probs) == 1.0, \
                'The sum of augmentation probabilities should equal to 1,' \
                ' but got {:.2f}'.format(sum(self.augment_probs))
        else:
            assert sum(self.augment_probs) <= 1.0, \
                'The sum of augmentation probabilities should be less than or ' \
                'equal to 1, but got {:.2f}'.format(sum(self.augment_probs))
            identity_prob = 1 - sum(self.augment_probs)
            if identity_prob > 0:
                self.augments += [Identity(prob=identity_prob)]
                self.augment_probs += [identity_prob]

    def __call__(self, inputs):
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(0)
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(-1)
        random_state = np.random.RandomState(random.randint(0, 2**32 - 1))
        aug = random_state.choice(self.augments, p=self.augment_probs)
        return aug(inputs)