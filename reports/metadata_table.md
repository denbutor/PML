| feature | group | missing_share | rule | additive | partitioning |
|:---|:---|---:|:---|:---|:---|
| Tenure | numeric_continuous | 0.0473 | max_share: 0.10, median | log1p → standard_scaler | winsorize |
| WarehouseToHome | numeric_continuous | 0.0457 | max_share: 0.10, median | log1p → standard_scaler | winsorize |
| HourSpendOnApp | numeric_continuous | 0.0437 | max_share: 0.10, median | log1p → standard_scaler | winsorize |
| NumberOfDeviceRegistered | numeric_continuous | 0.0000 | max_share: 0.10, median | log1p → standard_scaler | winsorize |
| NumberOfAddress | numeric_continuous | 0.0000 | max_share: 0.10, median | log1p → standard_scaler | winsorize |
| OrderAmountHikeFromlastYear | numeric_continuous | 0.0462 | max_share: 0.10, median | log1p → standard_scaler | winsorize |
| CouponUsed | numeric_continuous | 0.0462 | max_share: 0.10, median | log1p → standard_scaler | winsorize |
| OrderCount | numeric_continuous | 0.0471 | max_share: 0.10, median | log1p → standard_scaler | winsorize |
| DaySinceLastOrder | numeric_continuous | 0.0548 | max_share: 0.10, median | log1p → standard_scaler | winsorize |
| CashbackAmount | numeric_continuous | 0.2016 | max_share: 0.40, median + indicator | log1p → standard_scaler | winsorize |
| Complain | numeric_binary | 0.0000 | max_share: 0.10, median | standard_scaler | passthrough |
| PreferredLoginDevice | categorical_nominal | 0.0000 | max_share: 0.10, most_frequent | woe | one_hot |
| PreferredPaymentMode | categorical_nominal | 0.0000 | max_share: 0.10, most_frequent | woe | one_hot |
| PreferedOrderCat | categorical_nominal | 0.1499 | max_share: 1.00, constant | woe | one_hot |
| MaritalStatus | categorical_nominal | 0.0000 | max_share: 0.10, most_frequent | woe | one_hot |
| SatisfactionScore | categorical_ordinal | 0.0000 | max_share: 0.10, most_frequent | ordinal_encoder → standard_scaler | ordinal_encoder |