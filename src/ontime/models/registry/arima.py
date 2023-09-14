from typing import Optional, Tuple

from darts.models.forecasting.arima import ARIMA

from ...abstract import AbstractBaseModel


class ARIMA(ARIMA, AbstractBaseModel):
    def __init__(
        self,
        p: int = 12,
        d: int = 1,
        q: int = 0,
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
        random_state: Optional[int] = None,
        add_encoders: Optional[dict] = None,
    ):
        super().__init__(p, d, q, seasonal_order, trend, random_state, add_encoders)
