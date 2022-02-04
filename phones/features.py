from unicodedata import normalize
import pandas as pd
import numpy as np

class Features():
    def __init__(self, feature_type="phoible"):
        if feature_type == "phoible":
            self.data = pd.read_csv(
                "https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv"
            )

        _df1 = pd.read_csv(
            "https://raw.githubusercontent.com"
            + "/dmort27/panphon/master/panphon/data/ipa_all.csv"
        )
        _df2 = pd.read_csv(
            "https://raw.githubusercontent.com"
            + "/dmort27/panphon/master/panphon/data/ipa_bases.csv"
        )
        ipa_weights_df = pd.read_csv(
            "https://raw.githubusercontent.com"
            + "/dmort27/panphon/master/panphon/data/feature_weights.csv"
        )
