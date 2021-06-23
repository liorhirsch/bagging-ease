import numpy as np
import pandas as pd
from os.path import join
from glob import glob
from TorchEASE.src.main.EASE import TorchEASE

data_path = 'data/msd/pro_sg'

train_df = pd.read_csv(join(data_path, f"train.csv"))

val_df_tr = pd.read_csv(join(data_path, f"validation_tr.csv"))
val_df_te = pd.read_csv(join(data_path, f"validation_te.csv"))

test_df_tr = pd.read_csv(join(data_path, f"test_tr.csv"))
test_df_te = pd.read_csv(join(data_path, f"test_te.csv"))

te_implicit = TorchEASE(train_df, user_col="uid", item_col="sid")
te_implicit.fit()

predictions = te_implicit.predict(val_df_tr, k=20)