"""
Neural ODE Utilities for GNODEVAE

This module provides utility functions and classes for integrating Neural Ordinary
Differential Equations (ODEs) into the GNODEVAE framework. Neural ODEs enable
modeling of continuous-time dynamics in the latent space, which is particularly
useful for trajectory inference in single-cell data.

Key components:
- get_step_size: Compute step size for ODE solvers
- LatentODEfunc: Neural network that defines the ODE dynamics
"""

import torch  
import torch.nn as nn  
from typing import Optional


def get_step_size(step_size: Optional[float], t1: float, t2: float, t_size: int) -> dict:  
    """
    Compute the step size options for ODE solvers.
    
    This function calculates the appropriate step size for numerical ODE solvers
    based on the time range and number of time steps. The step size controls
    the granularity of the numerical integration.
    
    Parameters
    ----------
    step_size : Optional[float]
        The desired step size multiplier. If None, no step size constraints are set,
        and the solver uses adaptive stepping.
    t1 : float
        The start time of the ODE integration.
    t2 : float
        The end time of the ODE integration.
    t_size : int
        The number of time steps for integration.
        
    Returns
    -------
    dict
        A dictionary containing the step size options for the ODE solver.
        Empty dict if step_size is None, otherwise contains {"step_size": computed_value}.
        
    Examples
    --------
    >>> get_step_size(0.1, 0.0, 1.0, 10)
    {'step_size': 0.01}
    >>> get_step_size(None, 0.0, 1.0, 10)
    {}
    """
    if step_size is None:  
        return {}  

    # Compute normalized step size based on time range and number of steps
    computed_step_size = (t2 - t1) / t_size / step_size  
    return {"step_size": computed_step_size}  


class LatentODEfunc(nn.Module):  
    """
    Neural network that defines the dynamics of the latent ODE system.
    
    This class implements a neural network that models how the latent state
    changes over time. It takes a latent state and returns its derivative,
    which is used by ODE solvers to integrate trajectories. The network uses
    a simple two-layer architecture with ELU activation.
    
    In the context of single-cell analysis, this models how cells move through
    the latent space over developmental time or other continuous processes.
    
    Parameters
    ----------
    n_latent : int, default=5
        The dimensionality of the latent space (number of latent features).
    n_hidden : int, default=25
        The dimensionality of the hidden layer. Larger values provide more
        capacity but may lead to overfitting.
        
    Attributes
    ----------
    fc1 : nn.Linear
        First fully connected layer (latent_dim -> hidden_dim)
    elu : nn.ELU
        ELU activation function for smooth gradients
    fc2 : nn.Linear
        Second fully connected layer (hidden_dim -> latent_dim)
        
    Notes
    -----
    The ELU activation is chosen for its smoothness, which helps with
    stable ODE integration. The output dimension matches the input dimension
    since we're computing derivatives in the latent space.
    
    Examples
    --------
    >>> ode_func = LatentODEfunc(n_latent=10, n_hidden=50)
    >>> t = torch.tensor(0.5)
    >>> z = torch.randn(10)
    >>> dz_dt = ode_func(t, z)  # Compute derivative
    """

    def __init__(self, n_latent: int = 5, n_hidden: int = 25):  
        super().__init__()  
        self.fc1 = nn.Linear(n_latent, n_hidden)  
        self.elu = nn.ELU()  
        self.fc2 = nn.Linear(n_hidden, n_latent)  

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  
        """
        Compute the derivative of the latent state at a given time.
        
        This function defines the ODE: dz/dt = f(z, t), where z is the latent state.
        Note that while time `t` is passed for compatibility with ODE solvers,
        this implementation uses a time-invariant (autonomous) ODE where the
        dynamics don't explicitly depend on t.
        
        Parameters
        ----------
        t : torch.Tensor
            Current time point. Included for compatibility with ODE solvers
            but not used in this time-invariant implementation.
        x : torch.Tensor
            Current latent state, shape (batch_size, n_latent) or (n_latent,).
            
        Returns
        -------
        torch.Tensor
            Derivative of the latent state (dz/dt), same shape as input x.
            
        Notes
        -----
        The ODE solver will use these derivatives to integrate the trajectory:
        z(t) = z(t0) + ∫[t0 to t] f(z(τ), τ) dτ
        """
        out = self.fc1(x)  
        out = self.elu(out)  
        out = self.fc2(out)  
        return out