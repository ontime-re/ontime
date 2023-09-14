from typing import Optional, Tuple, Union, List

from darts.models.forecasting.catboost_model import CatBoostModel

from ...abstract import AbstractBaseModel


class CatBoost(CatBoostModel, AbstractBaseModel):
    def __init__(
        self,
        lags: Union[int, list] = None,
        lags_past_covariates: Union[int, List[int]] = None,
        lags_future_covariates: Union[Tuple[int, int], List[int]] = None,
        output_chunk_length: int = 1,
        add_encoders: Optional[dict] = None,
        likelihood: str = None,
        quantiles: List = None,
        random_state: Optional[int] = None,
        multi_models: Optional[bool] = True,
        use_static_covariates: bool = True,
        **kwargs
    ):
        super().__init__(
            lags,
            lags_past_covariates,
            lags_future_covariates,
            output_chunk_length,
            add_encoders,
            likelihood,
            quantiles,
            random_state,
            multi_models,
            use_static_covariates,
            **kwargs
        )
