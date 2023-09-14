from typing import Optional, Tuple, Union, List

from darts.models.forecasting.tcn_model import TCNModel

from ...abstract import AbstractBaseModel


class TCN(TCNModel, AbstractBaseModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        kernel_size: int = 3,
        num_filters: int = 3,
        num_layers: Optional[int] = None,
        dilation_base: int = 2,
        weight_norm: bool = False,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(
            input_chunk_length,
            output_chunk_length,
            kernel_size,
            num_filters,
            num_layers,
            dilation_base,
            weight_norm,
            dropout,
            **kwargs
        )
