from functools import lru_cache
import hashlib
import os
from typing import Callable, Dict, Iterable, List, Optional, Union
from unicodedata import normalize
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.spatial import distance

from .sources import PhoneSource, PHOIBLE


class Phone:
    def __init__(
        self,
        index: str,
        features: Dict[str, Union[int, str]],
        language_code: Optional[str] = None,
        allophones: Optional[List[str]] = None,
    ) -> None:
        """Create a new `Phone` object.
        
        Args:
            index: The index of the phone in the phone set.
            features: A dictionary of features. When not provided with numerical values, `-` will be replaced with `-1` and `+` with `1`.
            language_code: The language code of the language that the phoneme belongs to.
            allophones: A list of allophones for the phoneme.
        """
        self.index = index
        self.feature_names = sorted(features.keys())
        self.vector = [features[k] for k in self.feature_names]
        self.vector = np.array(self.vector)
        self.language_code = language_code
        if not isinstance(allophones, list):
            try:
                self.allophones = [a for a in allophones.split() if a != index]
            except AttributeError:
                # NaN
                self.allophones = []
        else:
            self.allophones = allophones


class PhoneCollection:
    def __init__(
        self,
        source: PhoneSource = PHOIBLE,
        cache_dir: str = "$HOME/.cache/phones",
        merge_same_language: bool = True,
        merge_same_phone: bool = False,
        drop_dialects: bool = True,
    ) -> None:
        """Creates a ``PhoneCollection`` object that loads phones from a ``PhoneSource`` into a pandas DataFrame.
        
        Args:
            source: The ``PhoneSource`` object that defines the source of the data.
            cache_dir: The directory where the data will be downloaded and cached.
            merge_same_language: If true, multiple phone definitions in the same language are merged.
            drop_dialects: If true, dialects are ignored.
        """
        self.source = source
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        dfs = []
        for url in source.urls:
            url_hash = hashlib.sha224(url.encode()).hexdigest()
            download_path = os.path.join(cache_dir, url_hash + ".pkl")
            if os.path.isfile(download_path):
                df = pd.read_pickle(download_path)
            else:
                df = pd.read_csv(url, dtype=str)
                df.to_pickle(download_path)
            dfs.append(df)
        self.data = pd.concat(dfs)
        assert merge_same_language != merge_same_phone
        for feature in self.source.feature_columns:
            self.data[feature] = self.data[feature].apply(
                PhoneCollection.feature_to_weight
            )
        if drop_dialects and self.source.dialect_column is not None:
            self.data = self.data[self.data[self.source.dialect_column].isna()]
        if merge_same_language and self.source.language_column is not None:
            self.data = self.data.dropna(subset=[self.source.language_column])
            cols = [
                self.source.language_column,
                self.source.index_column,
                self.source.allophone_column,
            ]
            if not drop_dialects and self.source.dialect_column is not None:
                cols.append(self.source.dialect_column)
            self.data = (
                self.data.groupby(cols)[self.source.feature_columns]
                .mean()
                .reset_index()
            )

    @staticmethod
    def feature_to_weight(feature):
        if isinstance(feature, str):
            try:
                return float(feature.replace("-", "-1").replace("+", "1"))
            except ValueError:
                if "," in feature:
                    return np.mean(
                        [
                            float(f.replace("-", "-1").replace("+", "1"))
                            for f in feature.split(",")
                        ]
                    )
                else:
                    return 0.0

    def _row_to_phone(self, row):
        idx = row[self.source.index_column]
        features = {f: row[f] for f in self.source.feature_columns if row[f] != "N"}
        language = None
        if self.source.language_column is not None:
            language = row[self.source.language_column]
        allophones = None
        if self.source.allophone_column is not None:
            allophones = row[self.source.allophone_column]
        if len(features) > 0:
            return Phone(idx, features, language, allophones)
        return None

    def get_phone_inventory(self, language: str):
        phone_df = (
            self.data.groupby(
                [
                    self.source.language_column,
                    self.source.index_column,
                    self.source.allophone_column,
                ]
            )
            .mean()
            .reset_index()
        )
        phone_df = phone_df[phone_df[self.source.language_column] == language]
        return phone_df

    def get_mean_allophone_distance(self, language_subset=None, distance_weights=None):
        dists = []
        if language_subset is not None:
            data = self.data[self.data[self.source.language_column].isin(language_subset)]
        else:
            data = self.data
        for _, row in tqdm(data.iterrows(), total=len(data)):
            phone = row[self.source.index_column]
            for allophone in row[self.source.allophone_column].split():
                if allophone != phone:
                    try:
                        dists.append(self.get_mean_phone_distance(allophone, phone, distance_weights=distance_weights))
                    except ValueError:
                        for subphone in allophone:
                            try:
                                dists.append(self.get_mean_phone_distance(subphone, phone, distance_weights=distance_weights))
                            except ValueError:
                                pass
        return np.mean(dists)

    @lru_cache(maxsize=1024)
    def get_mean_phone_distance(
        self,
        phone: str,
        other_phone: str,
        distance_fn: Callable[
            [Iterable[float], Iterable[float]], float
        ] = distance.euclidean,
        distance_weights = None
    ) -> float:
        phones_df = self.data.groupby(self.source.index_column).mean().reset_index()
        phone1_df = phones_df[phones_df[self.source.index_column]==phone][self.source.feature_columns].values
        phone2_df = phones_df[phones_df[self.source.index_column]==other_phone][self.source.feature_columns].values
        if distance_weights is None:
            return distance_fn(phone1_df, phone2_df)
        else:
            distance_weights = np.array(distance_weights)
            distance_weights = distance_weights / distance_weights.sum() * len(distance_weights)
            return distance_fn(phone1_df * distance_weights, phone2_df * distance_weights)

    def get_closest(
        self,
        phone: str,
        src_language: str,
        tgt_language: str,
        distance_fn: Callable[
            [Iterable[float], Iterable[float]], float
        ] = distance.euclidean,
        distance_weights=None,
        allow_allophones=True,
        return_all=False,
    ) -> List[Phone]:
        assert self.source.language_column is not None
        phone = normalize("NFC", phone)
        src_phone_df = self.get_phone_inventory(src_language)
        src_phone_vec = src_phone_df[src_phone_df[self.source.index_column] == phone][
            self.source.feature_columns
        ]
        if len(src_phone_vec) == 0 and allow_allophones:
            src_phone_vec = src_phone_df[
                src_phone_df[self.source.allophone_column].apply(
                    lambda x: phone in x.split()
                )
            ][self.source.feature_columns]
        if len(src_phone_vec) > 0:
            src_phone_vec = np.mean(src_phone_vec, axis=0)
        tgt_phone_df = self.get_phone_inventory(tgt_language)
        smallest_dist = float("inf")
        results = []
        allophones = []
        for tgt_phone in tgt_phone_df[self.source.index_column].unique():
            rows = tgt_phone_df[tgt_phone_df[self.source.index_column] == tgt_phone]
            tgt_phone_vec = rows[self.source.feature_columns].values
            if len(tgt_phone_vec) > 0:
                tgt_phone_vec = np.mean(tgt_phone_vec, axis=0)
            if distance_weights is None:
                distance_weights = np.ones(len(self.source.feature_columns))
                distance_weights = distance_weights / distance_weights.sum() * len(distance_weights)
            dist = distance_fn(src_phone_vec * distance_weights, tgt_phone_vec * distance_weights)
            for _, row in rows.iterrows():
                if phone in row[self.source.allophone_column].split():
                    allophones.append((dist, phone))
                    break
            if dist <= smallest_dist or return_all:
                if dist < smallest_dist:
                    smallest_dist = dist
                    if not return_all:
                        results = []
                results.append((dist, row[self.source.index_column]))

        return {
            "closest": sorted(results),
            "allophones": allophones,
        }
