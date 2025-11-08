from datetime import timedelta
from feast import Entity, FileSource, FeatureView, Field
from feast.types import Float32
from feast.value_type import ValueType

# Entity
stock = Entity(
    name="stock_id",
    join_keys=["stock_id"],
    value_type=ValueType.STRING,
)

# Parquet source (note: NO file_format arg needed)
stock_file_source_v0 = FileSource(
    path="../data/processed/v0+v1/train.parquet",
    timestamp_field="timestamp",
)

# Feature view
stock_features_view = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=365),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),
    ],
    source=stock_file_source_v0,
    online=True,
)

