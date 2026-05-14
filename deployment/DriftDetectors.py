# All credits for this file to the wonderful Alibi Detect library: https://github.com/SeldonIO/alibi-detect/tree/master

import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union, Tuple, List, Iterable
from typing_extensions import Literal, TypeAlias
from enum import Enum
import logging
import warnings
import copy
from tqdm import tqdm
import numpy as np
try:
    import torch  # noqa
    has_pytorch = True
except ImportError:
    has_pytorch = False
from torch import nn
has_tensorflow = False
has_keops = False

__version__ = "0.13.0.dev0"

TorchDeviceType: TypeAlias = Optional[Union[Literal['cuda', 'gpu', 'cpu'], 'torch.device']]

logger = logging.getLogger(__name__)

class Framework(str, Enum):
    PYTORCH = 'pytorch'
    TENSORFLOW = 'tensorflow'
    KEOPS = 'keops'
    SKLEARN = 'sklearn'

ERROR_TYPES = {
    "prophet": 'prophet',
    "tensorflow_probability": 'tensorflow',
    "tensorflow": 'tensorflow',
    "keras": 'tensorflow',
    "torch": 'torch',
    "pytorch": 'torch',
    "keops": 'keops',
    "pykeops": 'keops',
}

# Map from backend name to boolean value indicating its presence
HAS_BACKEND = {
    'tensorflow': has_tensorflow,
    'pytorch': has_pytorch,
    'sklearn': True,
    'keops': has_keops,
}

# "Large artefacts" - to save memory these are skipped in _set_config(), but added back in get_config()
# Note: The current implementation assumes the artefact is stored as a class attribute, and as a config field under
# the same name. Refactoring will be required if this assumption is to be broken.
LARGE_ARTEFACTS = ['x_ref', 'c_ref', 'preprocess_fn']

DEFAULT_META: Dict = {
    "name": None,
    "online": None,  # true or false
    "data_type": None,  # tabular, image or time-series
    "version": None,
    "detector_type": None  # drift, outlier or adversarial
}

def _iter_to_str(iterable: Iterable[str]) -> str:
    """ Correctly format iterable of items to comma seperated sentence string."""
    items = [f'`{option}`' for option in iterable]
    last_item_str = f'{items[-1]}' if not items[:-1] else f' and {items[-1]}'
    return ', '.join(items[:-1]) + last_item_str

class BackendValidator:
    def __init__(self, backend_options: Dict[Optional[str], List[str]], construct_name: str):
        """Checks for required sets of backend options.

        Takes a dictionary of backends plus extra dependencies and generates correct error messages if they are unmet.

        Parameters
        ----------
        backend_options
            Dictionary from backend to list of dependencies that must be satisfied. The keys are the available options
            for the user and the values should be a list of dependencies that are checked via the `HAS_BACKEND` map
            defined in this module. An example of `backend_options` would be `{'tensorflow': ['tensorflow'], 'pytorch':
            ['pytorch'], None: []}`.This would mean `'tensorflow'`, `'pytorch'` or `None` are available backend options.
            If the user passes a different backend they will receive and error listing the correct backends. In
            addition, if one of the dependencies in the `backend_option` values is missing for the specified backend
            the validator will issue an error message telling the user what dependency bucket to install.
        construct_name
            Name of the object that has a set of backends we need to verify.
        """
        self.backend_options = backend_options
        self.construct_name = construct_name

    def verify_backend(self, backend: str):
        """Verifies backend choice.

        Verifies backend is implemented and that the correct dependencies are installed for the requested backend. If
        the backend is not implemented or a dependency is missing then an error is issued.

        Parameters
        ----------
        backend
            Choice of backend the user wishes to initialize the alibi-detect construct with. Must be one of the keys
            in the `self.backend_options` dictionary.

        Raises
        ------
        NotImplementedError
            If backend is not a member of `self.backend_options.keys()` a `NotImplementedError` is raised. Note `None`
            is a valid choice of backend if it is set as a key on `self.backend_options.keys()`. If a backend is not
            implemented for an alibi-detect object then it should not have a key on `self.backend_options`.
        ImportError
            If one of the dependencies in `self.backend_options[backend]` is missing then an ImportError will be thrown
            including a message informing the user how to install.
        """
        if backend not in self.backend_options:
            self._raise_implementation_error(backend)

        dependencies = self.backend_options[backend]
        missing_deps = []
        for dependency in dependencies:
            if not HAS_BACKEND[dependency]:
                missing_deps.append(dependency)

        if missing_deps:
            self._raise_import_error(missing_deps, backend)

    def _raise_import_error(self, missing_deps: List[str], backend: str):
        """Raises import error if backend choice has missing dependency."""

        optional_dependencies = list(ERROR_TYPES[missing_dep] for missing_dep in missing_deps)
        optional_dependencies.sort()
        missing_deps_str = _iter_to_str(missing_deps)
        error_msg = (f'{missing_deps_str} not installed. Cannot initialize and run {self.construct_name} '
                     f'with {backend} backend.')
        pip_msg = '' if not optional_dependencies else \
            (f'The necessary missing dependencies can be installed using '
             f'`pip install alibi-detect[{" ".join(optional_dependencies)}]`.')
        raise ImportError(f'{error_msg} {pip_msg}')

    def _raise_implementation_error(self, backend: str):
        """Raises NotImplementedError error if backend choice is not implemented."""

        backend_list = _iter_to_str(self.backend_options.keys())
        raise NotImplementedError(f"{backend} backend not implemented. Use one of {backend_list} instead.")
            

class StateMixin(ABC):
    """
    Utility class that provides methods to save and load stateful attributes to disk.
    """
    t: int
    online_state_keys: Tuple[str, ...]

    def _set_state_dir(self, dirpath: Union[str, os.PathLike]):
        """
        Set the directory path to store state in, and create an empty directory if it doesn't already exist.

        Parameters
        ----------
        dirpath
            The directory to save state file inside.
        """
        self.state_dir = Path(dirpath)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save_state(self, filepath: Union[str, os.PathLike]):
        """
        Save a detector's state to disk in order to generate a checkpoint.

        Parameters
        ----------
        filepath
            The directory to save state to.
        """
        self._set_state_dir(filepath)
        suffix = '.pt' if hasattr(self, 'backend') and self.backend == Framework.PYTORCH else '.npz'
        _save_state_dict(self, self.online_state_keys, self.state_dir.joinpath('state' + suffix))
        logger.info('Saved state for t={} to {}'.format(self.t, self.state_dir))

    def load_state(self, filepath: Union[str, os.PathLike]):
        """
        Load the detector's state from disk, in order to restart from a checkpoint previously generated with
        `save_state`.

        Parameters
        ----------
        filepath
            The directory to load state from.
        """
        self._set_state_dir(filepath)
        suffix = '.pt' if hasattr(self, 'backend') and self.backend == Framework.PYTORCH else '.npz'
        _load_state_dict(self, self.state_dir.joinpath('state' + suffix), raise_error=True)
        logger.info('State loaded for t={} from {}'.format(self.t, self.state_dir))


class DriftConfigMixin:
    """
    A mixin class containing methods related to a drift detector's configuration dictionary.
    """
    config: Optional[dict] = None

    def get_config(self) -> dict:  # TODO - move to BaseDetector once config save/load implemented for non-drift
        """
        Get the detector's configuration dictionary.

        Returns
        -------
        The detector's configuration dictionary.
        """
        if self.config is not None:
            # Get config (stored in top-level self)
            cfg = self.config
            # Add large artefacts back to config
            for key in LARGE_ARTEFACTS:
                if key in cfg and hasattr(self._nested_detector, key):
                    cfg[key] = getattr(self._nested_detector, key)
            # Set x_ref_preprocessed flag
            # If no preprocess_at_init, always true!
            preprocess_at_init = getattr(self._nested_detector, 'preprocess_at_init', True)
            cfg['x_ref_preprocessed'] = preprocess_at_init and self._nested_detector.preprocess_fn is not None
            return cfg
        else:
            raise NotImplementedError('Getting a config (or saving via a config file) is not yet implemented for this'
                                      'detector')

    @classmethod
    def from_config(cls, config: dict):
        """
        Instantiate a drift detector from a fully resolved (and validated) config dictionary.

        Parameters
        ----------
        config
            A config dictionary matching the schema's in :class:`~alibi_detect.saving.schemas`.
        """
        # Check for existing version_warning. meta is pop'd as don't want to pass as arg/kwarg
        meta = config.pop('meta', None)
        meta = {} if meta is None else meta  # Needed because pydantic sets meta=None if it is missing from the config
        version_warning = meta.pop('version_warning', False)
        # Init detector
        detector = cls(**config)
        # Add version_warning
        detector.meta['version_warning'] = version_warning  # type: ignore[attr-defined]
        detector.config['meta']['version_warning'] = version_warning
        return detector

    def _set_config(self, inputs: dict):  # TODO - move to BaseDetector once config save/load implemented for non-drift
        """
        Set a detectors `config` attribute upon detector instantiation.

        Large artefacts are overwritten with `None` in order to avoid memory duplication. They're added back into
        the config later on by `get_config()`.

        Parameters
        ----------
        inputs
            The inputs (args/kwargs) given to the detector at instantiation.
        """
        # Set config metadata
        name = self.__class__.__name__

        # Init config dict
        self.config = {
            'name': name,
            'meta': {
                'version': __version__,
            }
        }

        # args and kwargs
        pop_inputs = ['self', '__class__', '__len__', 'name', 'meta']
        [inputs.pop(k, None) for k in pop_inputs]

        # Overwrite any large artefacts with None to save memory. They'll be added back by get_config()
        for key in LARGE_ARTEFACTS:
            if key in inputs and hasattr(self._nested_detector, key):
                inputs[key] = None

        self.config.update(inputs)

    @property
    def _nested_detector(self):
        """
        The low-level nested detector.
        """
        detector = self._detector if hasattr(self, '_detector') else self
        detector = detector._detector if hasattr(detector, '_detector') else detector
        return detector

def _save_state_dict_pt(state_dict: dict, filepath: Path):
    """
    Utility function to save a detector's state dictionary to a filepath using `torch.save`.

    Parameters
    ----------
    state_dict
        The state dictionary to save.
    filepath
        Directory to save state dictionary to.
    """
    # Save to disk
    torch.save(state_dict, filepath)
    

def _load_state_dict_pt(filepath: Path) -> dict:
    """
    Utility function to load a detector's state dictionary from a filepath with `torch.load`.

    Parameters
    ----------
    filepath
        Directory to load state dictionary from.

    Returns
    -------
    The loaded state dictionary.
    """
    return torch.load(filepath, weights_only=False)
    

def _save_state_dict(detector: StateMixin, keys: tuple, filepath: Path):
    """
    Utility function to save a detector's state dictionary to a filepath.

    Parameters
    ----------
    detector
        The detector to extract state attributes from.
    keys
        Tuple of state dict keys to populate dictionary with.
    filepath
        The file to save state dictionary to.
    """
    # Construct state dictionary
    state_dict = {key: getattr(detector, key, None) for key in keys}
    # Save to disk
    if filepath.suffix == '.pt':
        _save_state_dict_pt(state_dict, filepath)
    else:
        np.savez(filepath, **state_dict)


def _load_state_dict(detector: StateMixin, filepath: Path, raise_error: bool = True):
    """
    Utility function to load a detector's state dictionary from a filepath, and update the detectors attributes with
    the values in the state dictionary.

    Parameters
    ----------
    detector
        The detector to update.
    filepath
        File to load state dictionary from.
    raise_error
        Whether to raise an error if a file is not found at `filepath`. Otherwise, raise a warning and skip loading.

    Returns
    -------
    None. The detector is updated inplace.
    """
    if filepath.is_file():
        if filepath.suffix == '.pt':
            state_dict = _load_state_dict_pt(filepath)
        else:
            state_dict = np.load(str(filepath))
        for key, value in state_dict.items():
            setattr(detector, key, value)
    else:
        if raise_error:
            raise FileNotFoundError('State file not found at {}.'.format(filepath))
        else:
            logger.warning('State file not found at {}. Skipping loading of state.'.format(filepath))
            
    
class BaseDetector(ABC):
    """Base class for outlier, adversarial and drift detection algorithms."""

    def __init__(self):
        self.meta = copy.deepcopy(DEFAULT_META)
        self.meta['name'] = self.__class__.__name__
        self.meta['version'] = __version__

    def __repr__(self):
        return self.__class__.__name__

    @property
    def meta(self) -> Dict:
        return self._meta

    @meta.setter
    def meta(self, value: Dict):
        if not isinstance(value, dict):
            raise TypeError('meta must be a dictionary')
        self._meta = value

    @abstractmethod
    def score(self, X: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass

def concept_drift_dict():
    data = {
        'is_drift': None,
        'distance': None,
        'p_val': None,
        'threshold': None
    }
    return copy.deepcopy({"data": data, "meta": DEFAULT_META})


class BaseMultiDriftOnline(BaseDetector, StateMixin):
    t: int = 0
    thresholds: np.ndarray
    backend: Literal['pytorch', 'tensorflow']
    online_state_keys: Tuple[str, ...]

    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_size: int,
            preprocess_fn: Optional[Callable] = None,
            x_ref_preprocessed: bool = False,
            n_bootstraps: int = 1000,
            verbose: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None,
    ) -> None:
        """
        Base class for multivariate online drift detectors.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        ert
            The expected run-time (ERT) in the absence of drift. For the multivariate detectors, the ERT is defined
            as the expected run-time from t=0.
        window_size
            The size of the sliding test-window used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ert.
        verbose
            Whether or not to print progress during configuration.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if ert is None:
            logger.warning('No expected run-time set for the drift threshold. Need to set it to detect data drift.')

        self.ert = ert
        self.fpr = 1 / ert
        self.window_size = window_size

        # x_ref preprocessing
        self.x_ref_preprocessed = x_ref_preprocessed
        if preprocess_fn is not None and not isinstance(preprocess_fn, Callable):  # type: ignore[arg-type]
            raise ValueError("`preprocess_fn` is not a valid Callable.")
        if not self.x_ref_preprocessed and preprocess_fn is not None:
            self.x_ref = preprocess_fn(x_ref)
        else:
            self.x_ref = x_ref

        # Other attributes
        self.preprocess_fn = preprocess_fn
        self.n = len(x_ref)
        self.n_bootstraps = n_bootstraps  # nb of samples used to estimate thresholds
        self.verbose = verbose

        # store input shape for save and load functionality
        self.input_shape = get_input_shape(input_shape, x_ref)

        # set metadata
        self.meta['detector_type'] = 'drift'
        self.meta['data_type'] = data_type
        self.meta['online'] = True

    @abstractmethod
    def _configure_thresholds(self):
        pass

    @abstractmethod
    def _configure_ref_subset(self):
        pass

    @abstractmethod
    def _update_state(self, x_t: Union[np.ndarray, 'torch.Tensor']):
        pass

    def _preprocess_xt(self, x_t: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Private method to preprocess a single test instance ready for _update_state.

        Parameters
        ----------
        x_t
            A single test instance to be preprocessed.

        Returns
        -------
        The preprocessed test instance `x_t`.
        """
        # preprocess if necessary
        if self.preprocess_fn is not None:
            x_t = x_t[None, :] if isinstance(x_t, np.ndarray) else [x_t]
            x_t = self.preprocess_fn(x_t)[0]
        return x_t[None, :]

    def get_threshold(self, t: int) -> float:
        """
        Return the threshold for timestep `t`.

        Parameters
        ----------
        t
            The timestep to return a threshold for.

        Returns
        -------
        The threshold at timestep `t`.
        """
        return self.thresholds[t] if t < self.window_size else self.thresholds[-1]

    def _initialise_state(self) -> None:
        """
        Initialise online state (the stateful attributes updated by `score` and `predict`).

        If a subclassed detector has additional online state, an additional `_initialise_state` should be defined,
        with a call to `super()._initialise_state()` included (see `LSDDDriftOnlineTorch._initialise_state()` for
        an example).
        """
        self.t = 0  # corresponds to a test set of ref data
        self.test_stats = np.array([])
        self.drift_preds = np.array([])

    def reset(self) -> None:
        """
        Deprecated reset method. This method will be repurposed or removed in the future. To reset the detector to
        its initial state (`t=0`) use :meth:`reset_state`.
        """
        self.reset_state()
        warnings.warn('This method is deprecated and will be removed/repurposed in the future. To reset the detector '
                      'to its initial state use `reset_state`.', DeprecationWarning)

    def reset_state(self) -> None:
        """
        Resets the detector to its initial state (`t=0`). This does not include reconfiguring thresholds.
        """
        self._initialise_state()

    def predict(self, x_t: Union[np.ndarray, Any], return_test_stat: bool = True,
                ) -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether the most recent window of data has drifted from the reference data.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window.
        return_test_stat
            Whether to return the test statistic and threshold.

        Returns
        -------
        Dictionary containing ``'meta'`` and ``'data'`` dictionaries.
            - ``'meta'`` has the model's metadata.
            - ``'data'`` contains the drift prediction and optionally the test-statistic and threshold.
        """
        # Compute test stat and check for drift
        test_stat = self.score(x_t)
        threshold = self.get_threshold(self.t)
        drift_pred = int(test_stat > threshold)

        self.test_stats = np.concatenate([self.test_stats, np.array([test_stat])])
        self.drift_preds = np.concatenate([self.drift_preds, np.array([drift_pred])])

        # populate drift dict
        cd = concept_drift_dict()
        cd['meta'] = self.meta
        cd['data']['is_drift'] = drift_pred
        cd['data']['time'] = self.t
        cd['data']['ert'] = self.ert
        if return_test_stat:
            cd['data']['test_stat'] = test_stat
            cd['data']['threshold'] = threshold

        return cd


def get_input_shape(shape: Optional[Tuple], x_ref: Union[np.ndarray, list]) -> Optional[Tuple]:
    """ Optionally infer shape from reference data. """
    if isinstance(shape, tuple):
        return shape
    elif hasattr(x_ref, 'shape'):
        return x_ref.shape[1:]
    else:
        logger.warning('Input shape could not be inferred. '
                       'If alibi_detect.models.tensorflow.embedding.TransformerEmbedding '
                       'is used as preprocessing step, a saved detector cannot be reinitialized.')
        return None


def sigma_median(x: torch.Tensor, y: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """
    Bandwidth estimation using the median heuristic :cite:t:`Gretton2012`.

    Parameters
    ----------
    x
        Tensor of instances with dimension [Nx, features].
    y
        Tensor of instances with dimension [Ny, features].
    dist
        Tensor with dimensions [Nx, Ny], containing the pairwise distances between `x` and `y`.

    Returns
    -------
    The computed bandwidth, `sigma`.
    """
    n = min(x.shape[0], y.shape[0])
    n = n if (x[:n] == y[:n]).all() and x.shape == y.shape else 0
    n_median = n + (np.prod(dist.shape) - n) // 2 - 1
    sigma = (.5 * dist.flatten().sort().values[int(n_median)].unsqueeze(dim=-1)) ** .5
    return sigma


@torch.jit.script
def squared_pairwise_distance(x: torch.Tensor, y: torch.Tensor, a_min: float = 1e-30) -> torch.Tensor:
    """
    PyTorch pairwise squared Euclidean distance between samples x and y.

    Parameters
    ----------
    x
        Batch of instances of shape [Nx, features].
    y
        Batch of instances of shape [Ny, features].
    a_min
        Lower bound to clip distance values.
    Returns
    -------
    Pairwise squared Euclidean distance [Nx, Ny].
    """
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    dist = torch.addmm(y2.transpose(-2, -1), x, y.transpose(-2, -1), alpha=-2).add_(x2)
    return dist.clamp_min_(a_min)



class GaussianRBF(nn.Module):
    def __init__(
        self,
        sigma: Optional[torch.Tensor] = None,
        init_sigma_fn: Optional[Callable] = None,
        trainable: bool = False
    ) -> None:
        """
        Gaussian RBF kernel: k(x,y) = exp(-(1/(2*sigma^2)||x-y||^2). A forward pass takes
        a batch of instances x [Nx, features] and y [Ny, features] and returns the kernel
        matrix [Nx, Ny].

        Parameters
        ----------
        sigma
            Bandwidth used for the kernel. Needn't be specified if being inferred or trained.
            Can pass multiple values to eval kernel with and then average.
        init_sigma_fn
            Function used to compute the bandwidth `sigma`. Used when `sigma` is to be inferred.
            The function's signature should match :py:func:`~alibi_detect.utils.pytorch.kernels.sigma_median`,
            meaning that it should take in the tensors `x`, `y` and `dist` and return `sigma`. If `None`, it is set to
            :func:`~alibi_detect.utils.pytorch.kernels.sigma_median`.
        trainable
            Whether or not to track gradients w.r.t. `sigma` to allow it to be trained.
        """
        super().__init__()
        init_sigma_fn = sigma_median if init_sigma_fn is None else init_sigma_fn
        self.config: Dict[str, Any] = {'sigma': sigma, 'trainable': trainable, 'init_sigma_fn': init_sigma_fn}
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.empty(1), requires_grad=trainable)
            self.init_required = True
        else:
            sigma = sigma.reshape(-1)  # [Ns,]
            self.log_sigma = nn.Parameter(sigma.log(), requires_grad=trainable)
            self.init_required = False
        self.init_sigma_fn = init_sigma_fn
        self.trainable = trainable

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor],
                infer_sigma: bool = False) -> torch.Tensor:

        x, y = torch.as_tensor(x), torch.as_tensor(y)
        dist = squared_pairwise_distance(x.flatten(1), y.flatten(1))  # [Nx, Ny]

        if infer_sigma or self.init_required:
            if self.trainable and infer_sigma:
                raise ValueError("Gradients cannot be computed w.r.t. an inferred sigma value")
            sigma = self.init_sigma_fn(x, y, dist)
            with torch.no_grad():
                self.log_sigma.copy_(sigma.log().clone())
            self.init_required = False

        gamma = 1. / (2. * self.sigma ** 2)   # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = torch.exp(- torch.cat([(g * dist)[None, :, :] for g in gamma], dim=0))  # [Ns, Nx, Ny]
        return kernel_mat.mean(dim=0)  # [Nx, Ny]

    def get_config(self) -> dict:
        """
        Returns a serializable config dict (excluding the input_sigma_fn, which is serialized in alibi_detect.saving).
        """
        cfg = self.config.copy()
        if isinstance(cfg['sigma'], torch.Tensor):
            cfg['sigma'] = cfg['sigma'].detach().cpu().numpy().tolist()
        cfg.update({'flavour': Framework.PYTORCH.value})
        return cfg

    @classmethod
    def from_config(cls, config):
        """
        Instantiates a kernel from a config dictionary.

        Parameters
        ----------
        config
            A kernel config dictionary.
        """
        config.pop('flavour')
        return cls(**config)
    

def get_device(device: TorchDeviceType = None) -> torch.device:
    """
    Instantiates a PyTorch device object.

    Parameters
    ----------
    device
        Either `None`, a str ('gpu', 'cuda' or 'cpu') indicating the device to choose, or an already instantiated device
        object. If `None`, the GPU is selected if it is detected, otherwise the CPU is used as a fallback.

    Returns
    -------
    The instantiated device object.
    """
    if isinstance(device, torch.device):  # Already a torch device
        return device
    else:  # Instantiate device
        if device is None or device.lower() in ['gpu', 'cuda']:
            torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch_device.type == 'cpu':
                logger.warning('No GPU detected, fall back on CPU.')
        else:
            torch_device = torch.device('cpu')
            if device.lower() != 'cpu':
                logger.warning('Requested device not recognised, fall back on CPU.')
    return torch_device


def zero_diag(mat: torch.Tensor) -> torch.Tensor:
    """
    Set the diagonal of a matrix to 0

    Parameters
    ----------
    mat
        A 2D square matrix

    Returns
    -------
    A 2D square matrix with zeros along the diagonal
    """
    return mat - torch.diag(mat.diag())


def quantile(sample: torch.Tensor, p: float, type: int = 7, sorted: bool = False) -> float:
    """
    Estimate a desired quantile of a univariate distribution from a vector of samples

    Parameters
    ----------
    sample
        A 1D vector of values
    p
        The desired quantile in (0,1)
    type
        The method for computing the quantile.
        See https://wikipedia.org/wiki/Quantile#Estimating_quantiles_from_a_sample
    sorted
        Whether or not the vector is already sorted into ascending order

    Returns
    -------
    An estimate of the quantile

    """
    N = len(sample)

    if len(sample.shape) != 1:
        raise ValueError("Quantile estimation only supports vectors of univariate samples.")
    if not 1/N <= p <= (N-1)/N:
        raise ValueError(f"The {p}-quantile should not be estimated using only {N} samples.")

    sorted_sample = sample if sorted else sample.sort().values

    if type == 6:
        h = (N+1)*p
    elif type == 7:
        h = (N-1)*p + 1
    elif type == 8:
        h = (N+1/3)*p + 1/3
    h_floor = int(h)
    quantile = sorted_sample[h_floor-1]
    if h_floor != h:
        quantile += (h - h_floor)*(sorted_sample[h_floor]-sorted_sample[h_floor-1])

    return float(quantile)

def permed_lsdds(
    k_all_c: torch.Tensor,
    x_perms: List[torch.Tensor],
    y_perms: List[torch.Tensor],
    H: torch.Tensor,
    H_lam_inv: Optional[torch.Tensor] = None,
    lam_rd_max: float = 0.2,
    return_unpermed: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compute LSDD estimates from kernel matrix across various ref and test window samples

    Parameters
    ----------
    k_all_c
        Kernel matrix of similarities between all samples and the kernel centers.
    x_perms
        List of B reference window index vectors
    y_perms
        List of B test window index vectors
    H
        Special (scaled) kernel matrix of similarities between kernel centers
    H_lam_inv
        Function of H corresponding to a particular regulariation parameter lambda.
        See Eqn 11 of Bu et al. (2017)
    lam_rd_max
        The maximum relative difference between two estimates of LSDD that the regularization parameter
        lambda is allowed to cause. Defaults to 0.2. Only relavent if H_lam_inv is not supplied.
    return_unpermed
        Whether or not to return value corresponding to unpermed order defined by k_all_c

    Returns
    -------
    Vector of B LSDD estimates for each permutation, H_lam_inv which may have been inferred, and optionally \
    the unpermed LSDD estimate.
    """

    # Compute (for each bootstrap) the average distance to each kernel center (Eqn 7)
    k_xc_perms = torch.stack([k_all_c[x_inds] for x_inds in x_perms], 0)
    k_yc_perms = torch.stack([k_all_c[y_inds] for y_inds in y_perms], 0)
    h_perms = k_xc_perms.mean(1) - k_yc_perms.mean(1)

    if H_lam_inv is None:
        # We perform the initialisation for multiple candidate lambda values and pick the largest
        # one for which the relative difference (RD) between two difference estimates is below lambda_rd_max.
        # See Appendix A
        candidate_lambdas = [1/(4**i) for i in range(10)]  # TODO: More principled selection
        H_plus_lams = torch.stack(
            [H+torch.eye(H.shape[0], device=H.device)*can_lam for can_lam in candidate_lambdas], 0
        )
        H_plus_lam_invs = torch.inverse(H_plus_lams)
        H_plus_lam_invs = H_plus_lam_invs.permute(1, 2, 0)  # put lambdas in final axis

        omegas = torch.einsum('jkl,bk->bjl', H_plus_lam_invs, h_perms)  # (Eqn 8)
        h_omegas = torch.einsum('bj,bjl->bl', h_perms, omegas)
        omega_H_omegas = torch.einsum('bkl,bkl->bl', torch.einsum('bjl,jk->bkl', omegas, H), omegas)
        rds = (1 - (omega_H_omegas/h_omegas)).mean(0)
        less_than_rd_inds = (rds < lam_rd_max).nonzero()
        if len(less_than_rd_inds) == 0:
            repeats = k_all_c.shape[0] - torch.unique(k_all_c, dim=0).shape[0]
            if repeats > 0:
                msg = "Too many repeat instances for LSDD-based detection. \
                Try using MMD-based detection instead"
            else:
                msg = "Unknown error. Try using MMD-based detection instead"
            raise ValueError(msg)
        lam_index = less_than_rd_inds[0]
        lam = candidate_lambdas[lam_index]
        logger.info(f"Using lambda value of {lam:.2g} with RD of {float(rds[lam_index]):.2g}")
        H_plus_lam_inv = H_plus_lam_invs[:, :, int(lam_index.item())]
        H_lam_inv = 2*H_plus_lam_inv - (H_plus_lam_inv.transpose(0, 1) @ H @ H_plus_lam_inv)  # (below Eqn 11)

    # Now to compute an LSDD estimate for each permutation
    lsdd_perms = (h_perms * (H_lam_inv @ h_perms.transpose(0, 1)).transpose(0, 1)).sum(-1)  # (Eqn 11)

    if return_unpermed:
        n_x = x_perms[0].shape[0]
        h = k_all_c[:n_x].mean(0) - k_all_c[n_x:].mean(0)
        lsdd_unpermed = (h[None, :] * (H_lam_inv @ h[:, None]).transpose(0, 1)).sum()
        return lsdd_perms, H_lam_inv, lsdd_unpermed
    else:
        return lsdd_perms, H_lam_inv


class MMDDriftOnlineTorch(BaseMultiDriftOnline):
    online_state_keys: tuple = ('t', 'test_stats', 'drift_preds', 'test_window', 'k_xy')

    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_size: int,
            preprocess_fn: Optional[Callable] = None,
            x_ref_preprocessed: bool = False,
            kernel: Callable = GaussianRBF,
            sigma: Optional[np.ndarray] = None,
            n_bootstraps: int = 1000,
            device: TorchDeviceType = None,
            verbose: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Online maximum Mean Discrepancy (MMD) data drift detector using preconfigured thresholds.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        ert
            The expected run-time (ERT) in the absence of drift. For the multivariate detectors, the ERT is defined
            as the expected run-time from t=0.
        window_size
            The size of the sliding test-window used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        kernel
            Kernel used for the MMD computation, defaults to Gaussian RBF kernel.
        sigma
            Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array.
            The kernel evaluation is then averaged over those bandwidths. If `sigma` is not specified, the 'median
            heuristic' is adopted whereby `sigma` is set as the median pairwise distance between reference samples.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ERT.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``. Only relevant for 'pytorch' backend.
        verbose
            Whether or not to print progress during configuration.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            ert=ert,
            window_size=window_size,
            preprocess_fn=preprocess_fn,
            x_ref_preprocessed=x_ref_preprocessed,
            n_bootstraps=n_bootstraps,
            verbose=verbose,
            input_shape=input_shape,
            data_type=data_type
        )
        self.backend = Framework.PYTORCH.value
        self.meta.update({'backend': self.backend})

        # set device
        self.device = get_device(device)

        # initialize kernel
        sigma = torch.from_numpy(sigma).to(self.device) if isinstance(sigma,  # type: ignore[assignment]
                                                                      np.ndarray) else None
        self.kernel = kernel(sigma) if kernel == GaussianRBF else kernel

        # compute kernel matrix for the reference data
        self.x_ref = torch.from_numpy(self.x_ref).to(self.device)
        self.k_xx = self.kernel(self.x_ref, self.x_ref, infer_sigma=(sigma is None))

        self._configure_thresholds()
        self._configure_ref_subset()  # self.initialise_state() called inside here

    def _initialise_state(self) -> None:
        """
        Initialise online state (the stateful attributes updated by `score` and `predict`). This method relies on
        attributes defined by `_configure_ref_subset`, hence must be called afterwards.
        """
        super()._initialise_state()
        self.test_window = self.x_ref[self.init_test_inds]
        self.k_xy = self.kernel(self.x_ref[self.ref_inds], self.test_window)

    def _configure_ref_subset(self):
        """
        Configure the reference data split. If the randomly selected split causes an initial detection, further splits
        are attempted.
        """
        etw_size = 2 * self.window_size - 1  # etw = extended test window
        rw_size = self.n - etw_size  # rw = ref-window
        # Make split and ensure it doesn't cause an initial detection
        mmd_init = None
        while mmd_init is None or mmd_init >= self.get_threshold(0):
            # Make split
            perm = torch.randperm(self.n)
            self.ref_inds, self.init_test_inds = perm[:rw_size], perm[-self.window_size:]
            # Compute initial mmd to check for initial detection
            self._initialise_state()  # to set self.test_window and self.k_xy
            self.k_xx_sub = self.k_xx[self.ref_inds][:, self.ref_inds]
            self.k_xx_sub_sum = zero_diag(self.k_xx_sub).sum() / (rw_size * (rw_size - 1))
            k_yy = self.kernel(self.test_window, self.test_window)
            mmd_init = (
                    self.k_xx_sub_sum +
                    zero_diag(k_yy).sum() / (self.window_size * (self.window_size - 1)) -
                    2 * self.k_xy.mean()
            )

    def _configure_thresholds(self):
        """
        Configure the test statistic thresholds via bootstrapping.
        """
        # Each bootstrap sample splits the reference samples into a sub-reference sample (x)
        # and an extended test window (y). The extended test window will be treated as W overlapping
        # test windows of size W (so 2W-1 test samples in total)

        w_size = self.window_size
        etw_size = 2 * w_size - 1  # etw = extended test window
        rw_size = self.n - etw_size  # rw = sub-ref window

        perms = [torch.randperm(self.n) for _ in range(self.n_bootstraps)]
        x_inds_all = [perm[:-etw_size] for perm in perms]
        y_inds_all = [perm[-etw_size:] for perm in perms]

        if self.verbose:
            print("Generating permutations of kernel matrix..")
        # Need to compute mmd for each bs for each of W overlapping windows
        # Most of the computation can be done once however
        # We avoid summing the rw_size^2 submatrix for each bootstrap sample by instead computing the full
        # sum once and then subtracting the relavent parts (k_xx_sum = k_full_sum - 2*k_xy_sum - k_yy_sum).
        # We also reduce computation of k_xy_sum from O(nW) to O(W) by caching column sums

        k_full_sum = zero_diag(self.k_xx).sum()
        k_xy_col_sums_all = [
            self.k_xx[x_inds][:, y_inds].sum(0) for x_inds, y_inds in
            (tqdm(zip(x_inds_all, y_inds_all), total=self.n_bootstraps) if self.verbose else
             zip(x_inds_all, y_inds_all))
        ]
        k_xx_sums_all = [(
                                 k_full_sum - zero_diag(self.k_xx[y_inds][:, y_inds]).sum() - 2 * k_xy_col_sums.sum()
                         ) / (rw_size * (rw_size - 1)) for y_inds, k_xy_col_sums in zip(y_inds_all, k_xy_col_sums_all)]
        k_xy_col_sums_all = [k_xy_col_sums / (rw_size * w_size) for k_xy_col_sums in k_xy_col_sums_all]

        # Now to iterate through the W overlapping windows
        thresholds = []
        p_bar = tqdm(range(w_size), "Computing thresholds") if self.verbose else range(w_size)
        for w in p_bar:
            y_inds_all_w = [y_inds[w:w + w_size] for y_inds in y_inds_all]  # test windows of size w_size
            mmds = [(
                    k_xx_sum +
                    zero_diag(self.k_xx[y_inds_w][:, y_inds_w]).sum() / (w_size * (w_size - 1)) -
                    2 * k_xy_col_sums[w:w + w_size].sum())
                    for k_xx_sum, y_inds_w, k_xy_col_sums in zip(k_xx_sums_all, y_inds_all_w, k_xy_col_sums_all)
                    ]
            mmds = torch.tensor(mmds)  # an mmd for each bootstrap sample

            # Now we discard all bootstrap samples for which mmd is in top (1/ert)% and record the thresholds
            thresholds.append(quantile(mmds, 1 - self.fpr))
            y_inds_all = [y_inds_all[i] for i in range(len(y_inds_all)) if mmds[i] < thresholds[-1]]
            k_xx_sums_all = [
                k_xx_sums_all[i] for i in range(len(k_xx_sums_all)) if mmds[i] < thresholds[-1]
            ]
            k_xy_col_sums_all = [
                k_xy_col_sums_all[i] for i in range(len(k_xy_col_sums_all)) if mmds[i] < thresholds[-1]
            ]

        self.thresholds = thresholds

    def _update_state(self, x_t: torch.Tensor):  # type: ignore[override]
        """
        Update online state based on the provided test instance.

        Parameters
        ----------
        x_t
            The test instance.
        """
        self.t += 1
        kernel_col = self.kernel(self.x_ref[self.ref_inds], x_t)
        self.test_window = torch.cat([self.test_window[(1 - self.window_size):], x_t], 0)
        self.k_xy = torch.cat([self.k_xy[:, (1 - self.window_size):], kernel_col], 1)

    def score(self, x_t: Union[np.ndarray, Any]) -> float:
        """
        Compute the test-statistic (squared MMD) between the reference window and test window.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window.

        Returns
        -------
        Squared MMD estimate between reference window and test window.
        """
        x_t = super()._preprocess_xt(x_t)
        x_t = torch.from_numpy(x_t).to(self.device)
        self._update_state(x_t)
        k_yy = self.kernel(self.test_window, self.test_window)
        mmd = (
                self.k_xx_sub_sum +
                zero_diag(k_yy).sum() / (self.window_size * (self.window_size - 1)) -
                2 * self.k_xy.mean()
        )
        return float(mmd.detach().cpu())
    

class MMDDriftOnline(DriftConfigMixin):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_size: int,
            backend: str = 'tensorflow',
            preprocess_fn: Optional[Callable] = None,
            x_ref_preprocessed: bool = False,
            kernel: Optional[Callable] = None,
            sigma: Optional[np.ndarray] = None,
            n_bootstraps: int = 1000,
            device: TorchDeviceType = None,
            verbose: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Online maximum Mean Discrepancy (MMD) data drift detector using preconfigured thresholds.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        ert
            The expected run-time (ERT) in the absence of drift. For the multivariate detectors, the ERT is defined
            as the expected run-time from t=0.
        window_size
            The size of the sliding test-window used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        backend
            Backend used for the MMD implementation and configuration.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        kernel
            Kernel used for the MMD computation, defaults to Gaussian RBF kernel.
        sigma
            Optionally set the GaussianRBF kernel bandwidth. Can also pass multiple bandwidth values as an array.
            The kernel evaluation is then averaged over those bandwidths. If `sigma` is not specified, the 'median
            heuristic' is adopted whereby `sigma` is set as the median pairwise distance between reference samples.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ERT.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``. Only relevant for 'pytorch' backend.
        verbose
            Whether or not to print progress during configuration.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        # Set config
        self._set_config(locals())

        backend = backend.lower()
        BackendValidator(
            backend_options={Framework.TENSORFLOW: [Framework.TENSORFLOW],
                             Framework.PYTORCH: [Framework.PYTORCH]},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        kwargs = locals()
        args = [kwargs['x_ref'], kwargs['ert'], kwargs['window_size']]
        pop_kwargs = ['self', 'x_ref', 'ert', 'window_size', 'backend', '__class__']
        [kwargs.pop(k, None) for k in pop_kwargs]

        if kernel is None:
            if backend == Framework.TENSORFLOW:
                raise NotImplementedError("Custom kernels are not currently supported for the TensorFlow implementation of MMDDriftOnline.")
            
            kwargs.update({'kernel': GaussianRBF})

        if backend == Framework.TENSORFLOW:
            kwargs.pop('device', None)
            raise NotImplementedError("The TensorFlow implementation of MMDDriftOnline is not currently available. Please use the PyTorch implementation by setting `backend='pytorch'`.")
        
        self._detector = MMDDriftOnlineTorch(*args, **kwargs)  # type: ignore
        self.meta = self._detector.meta

    @property
    def t(self):
        return self._detector.t

    @property
    def test_stats(self):
        return self._detector.test_stats

    @property
    def thresholds(self):
        return [self._detector.thresholds[min(s, self._detector.window_size-1)] for s in range(self.t)]

    def reset_state(self):
        """
        Resets the detector to its initial state (`t=0`). This does not include reconfiguring thresholds.
        """
        self._detector.reset_state()

    def predict(self, x_t: Union[np.ndarray, Any], return_test_stat: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether the most recent window of data has drifted from the reference data.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window.
        return_test_stat
            Whether to return the test statistic (squared MMD) and threshold.

        Returns
        -------
        Dictionary containing ``'meta'`` and ``'data'`` dictionaries.
            - ``'meta'`` has the model's metadata.
            - ``'data'`` contains the drift prediction and optionally the test-statistic and threshold.
        """
        return self._detector.predict(x_t, return_test_stat)

    def score(self, x_t: Union[np.ndarray, Any]) -> float:
        """
        Compute the test-statistic (squared MMD) between the reference window and test window.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window.

        Returns
        -------
        Squared MMD estimate between reference window and test window.
        """
        return self._detector.score(x_t)

    def save_state(self, filepath: Union[str, os.PathLike]):
        """
        Save a detector's state to disk in order to generate a checkpoint.

        Parameters
        ----------
        filepath
            The directory to save state to.
        """
        self._detector.save_state(filepath)

    def load_state(self, filepath: Union[str, os.PathLike]):
        """
        Load the detector's state from disk, in order to restart from a checkpoint previously generated with
        `save_state`.

        Parameters
        ----------
        filepath
            The directory to load state from.
        """
        self._detector.load_state(filepath)

    def get_config(self) -> dict:  # Needed due to self.x_ref being a torch.Tensor when backend='pytorch'
        """
        Get the detector's configuration dictionary.

        Returns
        -------
        The detector's configuration dictionary.
        """
        cfg = super().get_config()
        if cfg.get('backend') == 'pytorch':
            cfg['x_ref'] = cfg['x_ref'].detach().cpu().numpy()
        return cfg
    
class LSDDDriftOnlineTorch(BaseMultiDriftOnline):
    online_state_keys: tuple = ('t', 'test_stats', 'drift_preds', 'test_window', 'k_xtc')

    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_size: int,
            preprocess_fn: Optional[Callable] = None,
            x_ref_preprocessed: bool = False,
            sigma: Optional[np.ndarray] = None,
            n_bootstraps: int = 1000,
            n_kernel_centers: Optional[int] = None,
            lambda_rd_max: float = 0.2,
            device: TorchDeviceType = None,
            verbose: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Online least squares density difference (LSDD) data drift detector using preconfigured thresholds.
        Motivated by Bu et al. (2017): https://ieeexplore.ieee.org/abstract/document/7890493
        We have made modifications such that a desired ERT can be accurately targeted however.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        ert
            The expected run-time (ERT) in the absence of drift. For the multivariate detectors, the ERT is defined
            as the expected run-time from t=0.
        window_size
            The size of the sliding test-window used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        sigma
            Optionally set the bandwidth of the Gaussian kernel used in estimating the LSDD. Can also pass multiple
            bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. If `sigma`
            is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance
            between reference samples.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ert.
        n_kernel_centers
            The number of reference samples to use as centers in the Gaussian kernel model used to estimate LSDD.
            Defaults to 2*window_size.
        lambda_rd_max
            The maximum relative difference between two estimates of LSDD that the regularization parameter
            lambda is allowed to cause. Defaults to 0.2 as in the paper.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``. Only relevant for 'pytorch' backend.
        verbose
            Whether or not to print progress during configuration.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__(
            x_ref=x_ref,
            ert=ert,
            window_size=window_size,
            preprocess_fn=preprocess_fn,
            x_ref_preprocessed=x_ref_preprocessed,
            n_bootstraps=n_bootstraps,
            verbose=verbose,
            input_shape=input_shape,
            data_type=data_type
        )
        self.backend = Framework.PYTORCH.value
        self.meta.update({'backend': self.backend})
        self.n_kernel_centers = n_kernel_centers
        self.lambda_rd_max = lambda_rd_max

        # set device
        self.device = get_device(device)

        self._configure_normalization()

        # initialize kernel
        if sigma is None:
            x_ref = torch.from_numpy(self.x_ref).to(self.device)  # type: ignore[assignment]
            self.kernel = GaussianRBF()
            _ = self.kernel(x_ref, x_ref, infer_sigma=True)
        else:
            sigma = torch.from_numpy(sigma).to(self.device) if isinstance(sigma,  # type: ignore[assignment]
                                                                          np.ndarray) else None
            self.kernel = GaussianRBF(sigma)

        if self.n_kernel_centers is None:
            self.n_kernel_centers = 2 * window_size

        self._configure_kernel_centers()
        self._configure_thresholds()
        self._configure_ref_subset()  # self.initialise_state() called inside here

    def _configure_normalization(self, eps: float = 1e-12):
        """
        Configure the normalization functions used to normalize reference and test data to zero mean and unit variance.
        The reference data `x_ref` is also normalized here.
        """
        x_ref = torch.from_numpy(self.x_ref).to(self.device)
        x_ref_means = x_ref.mean(0)
        x_ref_stds = x_ref.std(0)
        self._normalize = lambda x: (x - x_ref_means) / (x_ref_stds + eps)
        self._unnormalize = lambda x: (torch.as_tensor(x) * (x_ref_stds + eps) + x_ref_means).cpu().numpy()
        self.x_ref = self._normalize(x_ref).cpu().numpy()

    def _configure_kernel_centers(self):
        "Set aside reference samples to act as kernel centers"
        perm = torch.randperm(self.n)
        self.c_inds, self.non_c_inds = perm[:self.n_kernel_centers], perm[self.n_kernel_centers:]
        self.kernel_centers = torch.from_numpy(self.x_ref[self.c_inds]).to(self.device)
        if np.unique(self.kernel_centers.cpu().numpy(), axis=0).shape[0] < self.n_kernel_centers:
            perturbation = (torch.randn(self.kernel_centers.shape) * 1e-6).to(self.device)
            self.kernel_centers = self.kernel_centers + perturbation
        self.x_ref_eff = torch.from_numpy(self.x_ref[self.non_c_inds]).to(self.device)  # the effective reference set
        self.k_xc = self.kernel(self.x_ref_eff, self.kernel_centers)

    def _configure_thresholds(self):
        """
        Configure the test statistic thresholds via bootstrapping.
        """
        # Each bootstrap sample splits the reference samples into a sub-reference sample (x)
        # and an extended test window (y). The extended test window will be treated as W overlapping
        # test windows of size W (so 2W-1 test samples in total)

        w_size = self.window_size
        etw_size = 2 * w_size - 1  # etw = extended test window
        nkc_size = self.n - self.n_kernel_centers  # nkc = non-kernel-centers
        rw_size = nkc_size - etw_size  # rw = ref-window

        perms = [torch.randperm(nkc_size) for _ in range(self.n_bootstraps)]
        x_inds_all = [perm[:rw_size] for perm in perms]
        y_inds_all = [perm[rw_size:] for perm in perms]

        # For stability in high dimensions we don't divide H by (pi*sigma^2)^(d/2)
        # Results in an alternative test-stat of LSDD*(pi*sigma^2)^(d/2). Same p-vals etc.
        H = GaussianRBF(np.sqrt(2.) * self.kernel.sigma)(self.kernel_centers, self.kernel_centers)

        # Compute lsdds for first test-window. We infer regularisation constant lambda here.
        y_inds_all_0 = [y_inds[:w_size] for y_inds in y_inds_all]
        lsdds_0, H_lam_inv = permed_lsdds(
            self.k_xc, x_inds_all, y_inds_all_0, H, lam_rd_max=self.lambda_rd_max,
        )

        # Can compute threshold for first window
        thresholds = [quantile(lsdds_0, 1 - self.fpr)]
        # And now to iterate through the other W-1 overlapping windows
        p_bar = tqdm(range(1, w_size), "Computing thresholds") if self.verbose else range(1, w_size)
        for w in p_bar:
            y_inds_all_w = [y_inds[w:(w + w_size)] for y_inds in y_inds_all]
            lsdds_w, _ = permed_lsdds(self.k_xc, x_inds_all, y_inds_all_w, H, H_lam_inv=H_lam_inv)
            thresholds.append(quantile(lsdds_w, 1 - self.fpr))
            x_inds_all = [x_inds_all[i] for i in range(len(x_inds_all)) if lsdds_w[i] < thresholds[-1]]
            y_inds_all = [y_inds_all[i] for i in range(len(y_inds_all)) if lsdds_w[i] < thresholds[-1]]

        self.thresholds = thresholds
        self.H_lam_inv = H_lam_inv

    def _initialise_state(self) -> None:
        """
        Initialise online state (the stateful attributes updated by `score` and `predict`). This method relies on
        attributes defined by `_configure_ref_subset`, hence must be called afterwards.
        """
        super()._initialise_state()
        self.test_window = self.x_ref_eff[self.init_test_inds]
        self.k_xtc = self.kernel(self.test_window, self.kernel_centers)

    def _configure_ref_subset(self):
        """
        Configure the reference data split. If the randomly selected split causes an initial detection, further splits
        are attempted.
        """
        etw_size = 2 * self.window_size - 1  # etw = extended test window
        nkc_size = self.n - self.n_kernel_centers  # nkc = non-kernel-centers
        rw_size = nkc_size - etw_size  # rw = ref-window
        # Make split and ensure it doesn't cause an initial detection
        lsdd_init = None
        while lsdd_init is None or lsdd_init >= self.get_threshold(0):
            # Make split
            perm = torch.randperm(nkc_size)
            self.ref_inds, self.init_test_inds = perm[:rw_size], perm[-self.window_size:]
            # Compute initial lsdd to check for initial detection
            self._initialise_state()  # to set self.test_window and self.k_xtc
            self.c2s = self.k_xc[self.ref_inds].mean(0)  # (below Eqn 21)
            h_init = self.c2s - self.k_xtc.mean(0)  # (Eqn 21)
            lsdd_init = h_init[None, :] @ self.H_lam_inv @ h_init[:, None]  # (Eqn 11)

    def _update_state(self, x_t: torch.Tensor):  # type: ignore[override]
        """
        Update online state based on the provided test instance.

        Parameters
        ----------
        x_t
            The test instance.
        """
        self.t += 1
        k_xtc = self.kernel(x_t, self.kernel_centers)
        self.test_window = torch.cat([self.test_window[(1 - self.window_size):], x_t], 0)
        self.k_xtc = torch.cat([self.k_xtc[(1 - self.window_size):], k_xtc], 0)

    def score(self, x_t: Union[np.ndarray, Any]) -> float:
        """
        Compute the test-statistic (LSDD) between the reference window and test window.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window.

        Returns
        -------
        LSDD estimate between reference window and test window.
        """
        x_t = super()._preprocess_xt(x_t)
        x_t = torch.from_numpy(x_t).to(self.device)
        x_t = self._normalize(x_t)
        self._update_state(x_t)
        h = self.c2s - self.k_xtc.mean(0)  # (Eqn 21)
        lsdd = h[None, :] @ self.H_lam_inv @ h[:, None]  # (Eqn 11)
        return float(lsdd.detach().cpu())
    
    
class LSDDDriftOnline(DriftConfigMixin):
    def __init__(
            self,
            x_ref: Union[np.ndarray, list],
            ert: float,
            window_size: int,
            backend: str = 'tensorflow',
            preprocess_fn: Optional[Callable] = None,
            x_ref_preprocessed: bool = False,
            sigma: Optional[np.ndarray] = None,
            n_bootstraps: int = 1000,
            n_kernel_centers: Optional[int] = None,
            lambda_rd_max: float = 0.2,
            device: TorchDeviceType = None,
            verbose: bool = True,
            input_shape: Optional[tuple] = None,
            data_type: Optional[str] = None
    ) -> None:
        """
        Online least squares density difference (LSDD) data drift detector using preconfigured thresholds.
        Motivated by Bu et al. (2017): https://ieeexplore.ieee.org/abstract/document/7890493
        We have made modifications such that a desired ERT can be accurately targeted however.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        ert
            The expected run-time (ERT) in the absence of drift. For the multivariate detectors, the ERT is defined
            as the expected run-time from t=0.
        window_size
            The size of the sliding test-window used to compute the test-statistic.
            Smaller windows focus on responding quickly to severe drift, larger windows focus on
            ability to detect slight drift.
        backend
            Backend used for the LSDD implementation and configuration.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        x_ref_preprocessed
            Whether the given reference data `x_ref` has been preprocessed yet. If `x_ref_preprocessed=True`, only
            the test data `x` will be preprocessed at prediction time. If `x_ref_preprocessed=False`, the reference
            data will also be preprocessed.
        sigma
            Optionally set the bandwidth of the Gaussian kernel used in estimating the LSDD. Can also pass multiple
            bandwidth values as an array. The kernel evaluation is then averaged over those bandwidths. If `sigma`
            is not specified, the 'median heuristic' is adopted whereby `sigma` is set as the median pairwise distance
            between reference samples.
        n_bootstraps
            The number of bootstrap simulations used to configure the thresholds. The larger this is the
            more accurately the desired ERT will be targeted. Should ideally be at least an order of magnitude
            larger than the ert.
        n_kernel_centers
            The number of reference samples to use as centers in the Gaussian kernel model used to estimate LSDD.
            Defaults to 2*window_size.
        lambda_rd_max
            The maximum relative difference between two estimates of LSDD that the regularization parameter
            lambda is allowed to cause. Defaults to 0.2 as in the paper.
        device
            Device type used. The default tries to use the GPU and falls back on CPU if needed.
            Can be specified by passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of
            ``torch.device``. Only relevant for 'pytorch' backend.
        verbose
            Whether or not to print progress during configuration.
        input_shape
            Shape of input data.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        # Set config
        self._set_config(locals())

        backend = backend.lower()
        BackendValidator(
            backend_options={Framework.TENSORFLOW: [Framework.TENSORFLOW],
                             Framework.PYTORCH: [Framework.PYTORCH]},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        kwargs = locals()
        args = [kwargs['x_ref'], kwargs['ert'], kwargs['window_size']]
        pop_kwargs = ['self', 'x_ref', 'ert', 'window_size', 'backend', '__class__']
        [kwargs.pop(k, None) for k in pop_kwargs]

        if backend == Framework.TENSORFLOW:
            kwargs.pop('device', None)
            raise NotImplementedError("The TensorFlow implementation of LSDDDriftOnline is not currently available. Please use the PyTorch implementation by setting `backend='pytorch'`.")
        
        self._detector = LSDDDriftOnlineTorch(*args, **kwargs)  # type: ignore
        self.meta = self._detector.meta

    @property
    def t(self):
        return self._detector.t

    @property
    def test_stats(self):
        return self._detector.test_stats

    @property
    def thresholds(self):
        return [self._detector.thresholds[min(s, self._detector.window_size-1)] for s in range(self.t)]

    def reset_state(self):
        """
        Resets the detector to its initial state (`t=0`). This does not include reconfiguring thresholds.
        """
        self._detector.reset_state()

    def predict(self, x_t: Union[np.ndarray, Any], return_test_stat: bool = True) \
            -> Dict[Dict[str, str], Dict[str, Union[int, float]]]:
        """
        Predict whether the most recent window of data has drifted from the reference data.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window.
        return_test_stat
            Whether to return the test statistic (LSDD) and threshold.

        Returns
        -------
        Dictionary containing ``'meta'`` and ``'data'`` dictionaries.
            - ``'meta'`` has the model's metadata.
            - ``'data'`` contains the drift prediction and optionally the test-statistic and threshold.
        """
        return self._detector.predict(x_t, return_test_stat)

    def score(self, x_t: Union[np.ndarray, Any]) -> float:
        """
        Compute the test-statistic (LSDD) between the reference window and test window.

        Parameters
        ----------
        x_t
            A single instance to be added to the test-window.

        Returns
        -------
        LSDD estimate between reference window and test window.
        """
        return self._detector.score(x_t)

    def get_config(self) -> dict:  # Needed due to need to unnormalize x_ref
        """
        Get the detector's configuration dictionary.

        Returns
        -------
        The detector's configuration dictionary.
        """
        cfg = super().get_config()
        # Unnormalize x_ref
        cfg['x_ref'] = self._detector._unnormalize(cfg['x_ref'])
        return cfg

    def save_state(self, filepath: Union[str, os.PathLike]):
        """
        Save a detector's state to disk in order to generate a checkpoint.

        Parameters
        ----------
        filepath
            The directory to save state to.
        """
        self._detector.save_state(filepath)

    def load_state(self, filepath: Union[str, os.PathLike]):
        """
        Load the detector's state from disk, in order to restart from a checkpoint previously generated with
        :meth:`~save_state`.

        Parameters
        ----------
        filepath
            The directory to load state from.
        """
        self._detector.load_state(filepath)