import torch  
import torch.nn as nn  
from typing import Optional

def get_step_size(step_size: Optional[float], t1: float, t2: float, t_size: int) -> dict:  
    """  
    Compute the step size options for ODE solvers.  

    Parameters  
    ----------  
    step_size : Optional[float]  
        The desired step size. If None, no step size is set.  
    t1 : float  
        The start time.  
    t2 : float  
        The end time.  
    t_size : int  
        The number of time steps.  

    Returns  
    ----------  
    dict  
        A dictionary containing the step size options for the ODE solver.  
    """  
    if step_size is None:  
        return {}  

    computed_step_size = (t2 - t1) / t_size / step_size  
    return {"step_size": computed_step_size}  


class LatentODEfunc(nn.Module):  
    """  
    A class modeling the latent state derivatives with respect to time.  

    Parameters  
    ----------  
    n_latent : int, default=5  
        The dimensionality of the latent space.  
    n_hidden : int, default=25  
        The dimensionality of the hidden layer.  
    """  

    def __init__(self, n_latent: int = 5, n_hidden: int = 25):  
        super().__init__()  
        self.fc1 = nn.Linear(n_latent, n_hidden)  
        self.elu = nn.ELU()  
        self.fc2 = nn.Linear(n_hidden, n_latent)  

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  
        """  
        Compute the gradient at a given time `t` and a given state `x`.  

        Parameters  
        ----------  
        t : torch.Tensor  
            A given time point (not used in this implementation but included for compatibility with ODE solvers).  
        x : torch.Tensor  
            A given latent state.  

        Returns  
        ----------  
        torch.Tensor  
            The computed gradient of the latent state.  
        """  
        out = self.fc1(x)  
        out = self.elu(out)  
        out = self.fc2(out)  
        return out