"""Специфікація API на базі Pydantic v2: активовано сувору валідацію strict, встановлено режим extra="forbid", а допустимі діапазони у параметрах Field визначено на основі вимог бізнес-домену, а не за допомогою мінімальних і максимальних значень із навчального набору.
Чутливі атрибути (такі як CityTier чи Gender), а також унікальний ідентифікатор користувача повністю виключені зі структури запиту — передача будь-яких сторонніх даних автоматично повертає помилку з кодом 422.
Для показників, які містили природні пропуски ще на етапі ЛР1, передбачено тип Optional, де значення null сигналізує про відсутність інформації та згодом опрацьовується за допомогою AdaptiveImputer."""
from __future__ import annotations

from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

BATCH_LIMIT = 256


class PredictionRequest(BaseModel):
    """Дані клієнта на момент оцінки (знімок за останній місяць)."""

    model_config = ConfigDict(extra="forbid", strict=True, json_schema_extra={"example": {
        "Tenure": 4.0, "PreferredLoginDevice": "Mobile Phone", "WarehouseToHome": 15.0, "PreferredPaymentMode": "Debit Card",
        "HourSpendOnApp": 3.0, "NumberOfDeviceRegistered": 4, "PreferedOrderCat": "Laptop & Accessory",
        "SatisfactionScore": 3, "MaritalStatus": "Single", "NumberOfAddress": 3, "Complain": 1,
        "OrderAmountHikeFromlastYear": 15.0, "CouponUsed": 1.0, "OrderCount": 2.0, "DaySinceLastOrder": 3.0,
        "CashbackAmount": 150.0}})

    Tenure: Optional[float] = Field(ge=0, le=240, description="Стаж клієнта, місяців (до 20 років)")
    PreferredLoginDevice: Literal["Mobile Phone", "Computer"]
    WarehouseToHome: Optional[float] = Field(ge=0, le=300, description="Відстань склад → дім, км")
    PreferredPaymentMode: Literal["Debit Card", "Credit Card", "E wallet", "UPI", "Cash on Delivery"]
    HourSpendOnApp: Optional[float] = Field(ge=0, le=24, description="Годин у застосунку/на сайті")
    NumberOfDeviceRegistered: int = Field(ge=1, le=20)
    PreferedOrderCat: Optional[Literal["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"]]
    SatisfactionScore: int = Field(ge=1, le=5, description="Оцінка задоволеності 1–5")
    MaritalStatus: Literal["Single", "Married", "Divorced"]
    NumberOfAddress: int = Field(ge=1, le=50)
    Complain: Literal[0, 1] = Field(description="Скарга за останній місяць")
    OrderAmountHikeFromlastYear: Optional[float] = Field(ge=-100, le=1000, description="Зміна суми замовлень до минулого року, %")
    CouponUsed: Optional[float] = Field(ge=0, le=200, description="Використано купонів за місяць")
    OrderCount: Optional[float] = Field(ge=0, le=200, description="Замовлень за місяць")
    DaySinceLastOrder: Optional[float] = Field(ge=0, le=3650, description="Днів з останнього замовлення")
    CashbackAmount: Optional[float] = Field(ge=0, le=10000, description="Середній кешбек за місяць, у.о.")

    @model_validator(mode="after")
    def check_consistency(self):
        if self.CouponUsed is not None and self.OrderCount is not None and self.CouponUsed > self.OrderCount:
            raise ValueError("CouponUsed не може перевищувати OrderCount (купон використовується в замовленні)")
        return self


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    items: List[PredictionRequest] = Field(min_length=1, max_length=BATCH_LIMIT)


class Meta(BaseModel):
    model_version: str
    fallback_applied: bool
    ood_log_density: Optional[float]


class PredictionResponse(BaseModel):
    request_id: UUID
    score: float = Field(description="P(відтік) від моделі")
    label: Literal[0, 1] = Field(description="Рішення: 1 — запропонувати утримання (поріг ЛР3 або правило при fallback)")
    uncertainty: float = Field(description="Ентропія прогнозу H(p) ∈ [0, 1] (міра ЛР4A)")
    meta: Meta


class BatchResponse(BaseModel):
    items: List[PredictionResponse]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False)   # UUID приходить рядком
    request_id: UUID
    y_true: Literal[0, 1]


class FeedbackResponse(BaseModel):
    request_id: UUID
    stored: bool


class ShapBlock(BaseModel):
    values: List[float]
    base_value: float
    data: List[float]
    feature_names: List[str]


class Agreement(BaseModel):
    top5_overlap: float
    spearman: float


class ExplainResponse(BaseModel):
    prediction: PredictionResponse
    shap: ShapBlock
    lime: List[tuple[str, float]]
    agreement: Agreement


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[list] = None