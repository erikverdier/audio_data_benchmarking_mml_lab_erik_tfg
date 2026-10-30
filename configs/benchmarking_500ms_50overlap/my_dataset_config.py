from scripts.config.dataset_config import DatasetConfig

opensmile = DatasetConfig(
    name="my_dataset_opensmile_compare_2016_500ms_50overlap",
    data_type="features",
    feature_type="opensmile-compare-2016",
    data_dir="/Users/erikverdiermontero/Downloads/TFG_extraction/opensmile_500ms_50overlap",
    parquet_file="my_dataset_opensmile-compare-2016_0.5s.parquet",
    feature_column="embedding",
    label_column="vocal_technique",
    pad_sequences=False,
)