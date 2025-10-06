import numpy as np
import torch
from typing import Tuple, Optional
import inspect

from .postconditions import (
    LipschitzRobustnessPostcondition,
    Postcondition,
    StandardRobustnessPostcondition,
    OppositeFacesPostcondition,
    AlsomitraOutputPostcondition,
)
from .preconditions import (
    Precondition,
    EpsilonBall,
    AlsomitraProperty1,
    AlsomitraProperty2,
    AlsomitraProperty3,
    AlsomitraProperty4,
)
from property_driven_ml.logics import Logic, FuzzyLogic, STL, BooleanLogic

from abc import ABC, abstractmethod

BOOLEAN_LOGIC = BooleanLogic()


class Constraint(ABC):
    """
    Abstract base class for neural network property constraints, which are a combination of a precondition and postcondition.

    Provides a common interface for evaluating logical constraints on neural
    network outputs, supporting different logical frameworks.
    """

    @abstractmethod
    def __init__(
        self,
        device: torch.device,
        min: Optional[torch.Tensor] = None,
        max: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the constraint with the given device and parameters.
        The exact details of how pre and postconditions are initialized may vary
        depending on the specific constraint implementation.

        Args:
            device: PyTorch device for tensor computations.
            min: Optional minimum bound for input data.
            max: Optional maximum bound for input data.
            *args, **kwargs: Arguments needed to initialize precondition and postcondition.
        """
        self.min = min
        self.max = max
        self.device = device
        # Note: Concrete subclasses must set self.precondition and self.postcondition
        self.precondition: Precondition
        self.postcondition: Postcondition

    def uniform_sample(
        self,
        x: torch.Tensor,
        num_samples: int,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Generate uniform samples within the precondition region around input x.

        Args:
            x: Original input tensor.
            num_samples: Number of uniform samples to generate.
            *args, **kwargs: Additional arguments specific to the precondition.

        Returns:
            Tensor of shape (num_samples, *x.shape) containing uniform samples.
        """
        lo, hi = self.precondition.get_bounds(x, *args, **kwargs)
        if lo.isnan().any() or hi.isnan().any():
            if self.min is not None and self.max is not None:
                lo = torch.max(lo, self.min.to(self.device))
                hi = torch.min(hi, self.max.to(self.device))
            else:
                raise ValueError(
                    "Need to set min and max for unbounded dimensions in precondition"
                )
        # Expand lo and hi to shape (num_samples, *x.shape)
        # lo and hi should have same shape as x, so we add num_samples dimension

        target_shape = [num_samples] + list(x.shape)
        lo = lo.unsqueeze(0).expand(target_shape)
        hi = hi.unsqueeze(0).expand(target_shape)

        return torch.rand_like(lo) * (hi - lo) + lo

    def eval(
        self,
        N: torch.nn.Module,
        x: torch.Tensor,
        x_adv: torch.Tensor | None,
        y_target: torch.Tensor | None,
        logic: Logic,
        reduction: str | None = None,
        skip_sat: bool = False,
        postcondition_kwargs: dict = {},
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the constraint and compute loss and satisfaction.

        This method automatically adapts to any postcondition signature by using
        introspection to determine which parameters the postcondition needs and
        only passing those parameters.

        Examples of supported postcondition signatures:
            get_postcondition(self, N, x, x_adv)              # StandardRobustness
            get_postcondition(self, N, x_adv)                 # GroupConstraint and OppositeFacesConstraint
            get_postcondition(self, N, x_adv, scale, centre)  # AlsomitraOutput
            get_postcondition(self, N, x, x_adv, y_target)    # Future constraints

        Args:
            N: Neural network model.
            x: Original input tensor.
            x_adv: Adversarial input tensor, if None will use random example.
            y_target: Target output tensor.
            logic: Logic framework for constraint evaluation.
            reduction: Optional reduction method for loss aggregation.
            skip_sat: Whether to skip satisfaction computation.
            postcondition_args: Additional arguments to pass to get_postcondition
                                  (e.g., scale, centre for AlsomitraOutputConstraint).

        Returns:
            Tuple of (loss, satisfaction) tensors.
        """
        if x_adv is None:
            x_adv = self.uniform_sample(x, num_samples=1).squeeze(0)

        # Get the signature of the postcondition's get_postcondition method
        sig = inspect.signature(self.postcondition.get_postcondition)

        # Build a dictionary of all available parameters
        available_params = {
            "N": N,
            "x": x,
            "x_adv": x_adv,
            "y_target": y_target,
            **postcondition_kwargs,
        }

        # Filter to only include parameters that the method accepts
        method_params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue  # Skip 'self' parameter
            if param_name in available_params:
                method_params[param_name] = available_params[param_name]
            elif param.default is not param.empty:
                # Parameter has a default value, don't need to provide it
                continue
            else:
                # Required parameter not available - this could be an error
                # but we'll let the method call fail naturally with a clear error
                pass

        # Call the method with only the parameters it accepts
        postcondition = self.postcondition.get_postcondition(**method_params)

        loss = postcondition(logic)
        assert not torch.isnan(loss).any()  # nosec

        if isinstance(logic, FuzzyLogic):
            loss = torch.ones_like(loss) - loss
        elif isinstance(logic, STL):
            loss = torch.clamp(logic.NOT(loss), min=0.0)

        if skip_sat:
            # When skipping sat calculation, return a dummy tensor with same shape as loss
            sat = torch.zeros_like(loss)
        else:
            sat = postcondition(BOOLEAN_LOGIC).float()

        def agg(value: torch.Tensor) -> torch.Tensor:
            if reduction is None:
                return value
            elif reduction == "mean":
                # Convert boolean tensors to float for mean calculation
                if value.dtype == torch.bool:
                    value = value.float()
                return torch.mean(value)
            elif reduction == "sum":
                return torch.sum(value)
            else:
                raise ValueError(f"Unsupported reduction: {reduction}")

        return agg(loss), agg(sat)


class StandardRobustnessConstraint(Constraint):
    """Constraint ensuring model robustness to adversarial perturbations.

    Combines an epsilon ball precondition with a standard robustness postcondition.
    Enforces that the change in output probabilities between original and
    adversarial inputs remains within a specified threshold delta.
    """

    def __init__(
        self,
        device: torch.device,
        epsilon: float,
        delta: float = 0.05,
        std: Tuple[float, ...] | float | None = None,
    ):
        """Initialize StandardRobustnessConstraint.

        Args:
            device: PyTorch device for tensor computations.
            epsilon: Radius for epsilon ball precondition.
            delta: Threshold for robustness postcondition.
            std: Standard deviation for epsilon scaling.
        """
        super().__init__(device)
        self.precondition = EpsilonBall(device, epsilon, std)
        self.postcondition = StandardRobustnessPostcondition(device, delta)


class LipschitzRobustnessConstraint(Constraint):
    """Constraint ensuring Lipschitz robustness of the model."""

    def __init__(
        self,
        device: torch.device,
        epsilon: float = 0.01,
        L: float = 0.3,
    ):
        """Initialize LipschitzRobustnessConstraint.

        Args:
            device: PyTorch device for tensor computations.
            L: Lipschitz constant.
        """
        super().__init__(device)
        self.precondition = EpsilonBall(device, epsilon=epsilon)
        self.postcondition = LipschitzRobustnessPostcondition(device, L)


class OppositeFacesConstraint(Constraint):
    """Constraint ensuring a physical-world inspired constraint on dice images.

    Combines an epsilon ball precondition with a constraint on the outputs
    that enforces that the network may not predict faces at the same time that are
    on opposite sides of the die (e.g. faces 1 and 6).
    """

    def __init__(
        self,
        device: torch.device,
        epsilon: float = 24 / 255,
        delta: float = 1.0,
        std: Tuple[float, ...] | float | None = None,
    ):
        """Initialize OppositeFacesConstraint.

        Args:
            device: PyTorch device for tensor computations.
            epsilon: Radius for epsilon ball precondition.
            std: Standard deviation for epsilon scaling.
        """
        super().__init__(device)
        self.precondition = EpsilonBall(device, epsilon, std)
        self.postcondition = OppositeFacesPostcondition(device, delta)


class AlsomitraProperty1Constraint(Constraint):
    """Constraint for Alsomitra Property 1."""

    def __init__(
        self,
        device: torch.device,
        bounds_output: Tuple[float, float],
        min: torch.Tensor,
        max: torch.Tensor,
        normalize: bool = True,
        threshold: float = 2.0,
    ):
        """Initialize AlsomitraProperty1Constraint.

        Args:
            threshold: Threshold for the y output.
            bounds_output: Tuple specifying (min, max) bounds for the output y.
            min: Optional minimum bound for input data.
            max: Optional maximum bound for input data.
        """
        super().__init__(device=torch.device("cpu"), min=min, max=max)
        self.precondition = AlsomitraProperty1(threshold, min, max)
        self.postcondition = AlsomitraOutputPostcondition(
            device, lo=bounds_output[0], hi=bounds_output[1], normalize=normalize
        )


class AlsomitraProperty2Constraint(Constraint):
    """Constraint for Alsomitra Property 2."""

    def __init__(
        self,
        device: torch.device,
        min: torch.Tensor,
        max: torch.Tensor,
        y_threshold: float = 2.0,
        theta_threshold: Tuple[float, float] = (-0.786, 0.747),
        bounds_output: Tuple[Optional[float], Optional[float]] = (0.184, 0.19),
    ):
        """Initialize AlsomitraProperty2Constraint.

        Args:
            y_threshold: Threshold for the y output.
            theta_threshold: Tuple specifying (min, max) bounds for the theta output.
            min: Optional minimum bound for input data.
            max: Optional maximum bound for input data.
        """
        super().__init__(device=torch.device("cpu"), min=min, max=max)
        self.precondition = AlsomitraProperty2(y_threshold, theta_threshold, min, max)
        self.postcondition = AlsomitraOutputPostcondition(
            device, lo=bounds_output[0], hi=bounds_output[1], normalize=False
        )


class AlsomitraProperty3Constraint(Constraint):
    """Constraint for Alsomitra Property 3."""

    def __init__(
        self,
        device: torch.device,
        min: torch.Tensor,
        max: torch.Tensor,
        y_threshold: float = 2.0,
        v_y_threshold: float = -0.3,
        omega_threshold: float = -0.12,
        bounds_output: Tuple[Optional[float], Optional[float]] = (np.nan, 0.187),
    ):
        """Initialize AlsomitraProperty3Constraint.

        Args:
            y_threshold: Threshold for the y output.
            theta_threshold: Tuple specifying (min, max) bounds for the theta output.
            min: Optional minimum bound for input data.
            max: Optional maximum bound for input data.
        """
        super().__init__(device=torch.device("cpu"), min=min, max=max)
        self.precondition = AlsomitraProperty3(
            y_threshold, v_y_threshold, omega_threshold, min, max
        )
        self.postcondition = AlsomitraOutputPostcondition(
            device, lo=bounds_output[0], hi=bounds_output[1], normalize=False
        )


class AlsomitraProperty4Constraint(Constraint):
    """Constraint for Alsomitra Property 4."""

    def __init__(
        self,
        device: torch.device,
        min: torch.Tensor,
        max: torch.Tensor,
        y_threshold: float = 2.0,
        L: float = 0.3,
    ):
        """Initialize AlsomitraProperty4Constraint.

        Args:
            device: PyTorch device for tensor computations.
            y_threshold: Threshold for the y output.
            min: Optional minimum bound for input data.
            max: Optional maximum bound for input data.
        """
        super().__init__(device=torch.device("cpu"), min=min, max=max)
        self.precondition = AlsomitraProperty4(y_threshold, min, max)
        self.postcondition = LipschitzRobustnessPostcondition(device, L=L)
