from functools import lru_cache
import hashlib
from lib2to3.pgen2.token import OP
import os
from typing import Callable, Dict, Iterable, List, Tuple, Union
import unicodedata
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.spatial import distance
from copy import deepcopy
from pathlib import Path
import re

from .sources import PhoneSource, PHOIBLE
from .features import Phone

class PhoneCollection:
    def __init__(
        self,
        source: PhoneSource = PHOIBLE,
        cache_dir: str = f"{str(Path.home())}/.cache/phones",
        merge_same_language: bool = True,
        load_dialects: bool = False,
        _master: object = None,
    ) -> None:
        """Creates a ``PhoneCollection`` object that loads phones from a ``PhoneSource`` into a pandas DataFrame.

        Args:
            source: The ``PhoneSource`` object that defines the source of the data.
            cache_dir: The directory where the data will be downloaded and cached.
            merge_same_language: If true, multiple phone definitions in the same language are merged.
            load_dialects: If false, dialects are ignored.
        """
        self.source = source
        self.source.feature_columns = sorted(self.source.feature_columns)
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

        for feature in self.source.feature_columns:
            self.data[feature] = self.data[feature].apply(
                PhoneCollection.feature_to_weight
            )
        self.columns = [
            self.source.index_column,
            self.source.language_column,
            self.source.allophone_column,
        ]
        if not load_dialects and self.source.dialect_column is not None:
            self.data = self.data[self.data[self.source.dialect_column].isna()]
        else:
            self.columns.append(self.source.dialect_column)
        self.columns = [c for c in self.columns if c is not None]

        if merge_same_language and self.source.language_column is not None:
            self.data = self.data.dropna(subset=[self.source.language_column])

            self.data = (
                self.data.groupby(self.columns, dropna=False)[self.source.feature_columns]
                .mean()
                .reset_index()
            )

        if self.source.language_column is not None:
            self.data = self.data.dropna(subset=[self.source.index_column, self.source.language_column])
        else:
            self.data = self.data.dropna(subset=[self.source.index_column])

        self.data[self.source.index_column] = self.data[self.source.index_column].apply(
            lambda x: unicodedata.normalize("NFC", x)
        )
        if self.source.allophone_column is not None:
            self.data[self.source.allophone_column] = self.data[
                self.source.allophone_column
            ].apply(lambda x: unicodedata.normalize("NFC", str(x)))
        self.data = self.data[self.columns + self.source.feature_columns]
        self._master = deepcopy(self)
        self.lang_filter = None
        self.load_dialects = load_dialects

    @property
    def features(self):
        return self.source.feature_columns

    @staticmethod
    def feature_to_weight(feature: str) -> float:
        """
        If the feature is a string, try to convert it to a float "-" is converted to -1, "+" to 1.
        If it's a string but can't be converted to a float, return 0.0.
        If it's a comma-delimited list of "+" and "-", return the mean of the list of floats.

        Args:
            feature: The feature to be converted to a weight.

        Return:
            The string feature converted to a float in `[-1.,0.,1.]`
        """
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

    @property
    def lang_list(self) -> List[str]:
        return list(sorted(self.data[self.source.language_column].unique()))

    @property
    def phone_list(self) -> List[str]:
        return list(sorted(self.data[self.source.index_column].unique()))

    @property
    def dialect_list(self) -> List[str]:
        if not self.load_dialects:
            raise ValueError("Dialects are not loaded.")
        return list(sorted(self.data[self.source.dialect_column].dropna().unique()))

    def phones(self, phones: Union[str, List[str]]) -> object:
        """
        It takes a list of phones and returns a copy ``PhoneCollection`` with only the rows that have one of
        those phones.

        Args:
            phones: A list of phones or single phone to filter on.

        Returns:
            A new instance of the class, with the filtered data.
        """
        _self = deepcopy(self)
        if len(phones) > 0:
            if not isinstance(phones, list):
                phones = [phones]
            phones = [unicodedata.normalize("NFC", p) for p in phones]
            _self.data = _self.data[_self.data[self.source.index_column].isin(phones)]
        _self._master = deepcopy(self)
        return _self

    def dialects(self, dialects: Union[str, List[str], None], inplace=True) -> object:
        """
        It takes a list of dialects and returns a copy ``PhoneCollection`` with only the rows that have one of
        those dialects.

        Args:
            dialects: A list of dialects or single dialects to filter on. Use ```None``` to remove all dialects except the one without a specific name.
            inplace: Modifies the underlying dataframe, affecting phones.

        Returns:
            A new instance of the class, with the filtered data.
        """
        if not self.load_dialects:
            raise ValueError("Dialects are not loaded.")
        _self = deepcopy(self)
        if len(dialects) > 0:
            if not isinstance(dialects, list):
                dialects = [dialects]
            dialect_mask = _self.data[self.source.dialect_column].str.match("|".join([re.escape(d).lower() for d in dialects]), case=False)
            if dialects != [None]:
                dialect_mask = dialect_mask.fillna(False) # remove standard dialect
            else:
                dialect_mask = dialect_mask.isna()
            _self.data = _self.data[dialect_mask]
        _self.dialect_filter = set(dialects)
        if inplace:
            _self._master.data = _self.data
            _self._master.dialect_filter = _self.dialect_filter
        return _self

    def langs(self, langs, inplace=True) -> object:
        """
        It takes a list of languages and returns a copy ``PhoneCollection`` with only the rows that have one of
        those languages.

        Args:
            langs: A list of languages or single language to filter on.
            inplace: Modifies the underlying dataframe, affecting phones.

        Returns:
            A new instance of the class, with the filtered data.
        """
        _self = deepcopy(self)
        if len(langs) > 0:
            if not isinstance(langs, list):
                langs = [langs]
            _self.data = _self.data[_self.data[self.source.language_column].isin(langs)]
        _self.lang_filter = set(langs)
        if inplace:
            _self._master.data = _self.data
            _self._master.lang_filter = _self.lang_filter
        return _self

    @property
    def values(self) -> List[object]:
        """
        The collection as a list of phones.

        Returns:
            A list of ``Phone`` objects.
        """
        phone_df = (
            self.data.groupby(
                [c for c in self.columns if c != self.source.allophone_column]
            )
            .mean()
            .reset_index()
        )
        return [self._row_to_phone(row) for _, row in phone_df.iterrows()]

    @property
    def values_with_allophones(self) -> List[object]:
        """
        The collection as a list of phones.

        Returns:
            A list of ``Phone`` objects.
        """
        phone_df = (
            self.data.groupby(
                [c for c in self.columns]
            )
            .mean()
            .reset_index()
        )
        return [self._row_to_phone(row) for _, row in phone_df.iterrows()]

    @property
    def val(self) -> object:
        """
        If the collection is filtered down to a single phone, return that phone.

        Returns:
            A ``Phone`` object.
        """
        phone_df = self.data.groupby(self.columns).mean().reset_index()
        if len(phone_df) > 1:
            phone_df = self.data.groupby(self.source.index_column).mean().reset_index()
        results = [self._row_to_phone(row) for _, row in phone_df.iterrows()]
        assert len(results) == 1
        return results[0]

    def _get_phone_inventory(self, language: str):
        phone_df = self.data.groupby(self.columns).mean().reset_index()
        phone_df = phone_df[phone_df[self.source.language_column] == language]
        return phone_df

    def get_mean_allophone_distance(
        self, distance_weights=None, show_progress=False
    ) -> float:
        """
        For each row in the dataframe, we get the phone and allophone values.
        If the allophone is different from the phone, we get the mean distance between the allophone
        and the phone. We return the mean of all allophone <-> phone distances.

        Args:
            distance_weights: A dictionary of weights for each distance type.
            show_progress: If True, show a progress bar.

        Returns:
            The mean of the distances between allophones and their phones.
        """
        dists = []
        for _, row in tqdm(
            self.data.iterrows(), total=len(self.data), disable=(not show_progress)
        ):
            phone = row[self.source.index_column]
            for allophone in row[self.source.allophone_column].split():
                if allophone != phone:
                    try:
                        dists.append(
                            self.get_mean_phone_distance(
                                allophone, phone, distance_weights=distance_weights
                            )
                        )
                    except ValueError:
                        for subphone in allophone:
                            try:
                                dists.append(
                                    self.get_mean_phone_distance(
                                        subphone,
                                        phone,
                                        distance_weights=distance_weights,
                                    )
                                )
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
        distance_weights=None,
    ) -> float:
        """
        For a given phone, find the mean of all the features for that phone. Then, find the
        distance between that phone and another phone.

        Args:
            phone: The phone to compare to the other phone.
            other_phone: The other phone to compare to.
            distance_fn: The distance function to use.
            distance_weights: This is a list of weights for each feature.

        Return:
            The mean distance between the two phones.
        """
        phones_df = self.data.groupby(self.source.index_column).mean().reset_index()
        phone1_df = phones_df[phones_df[self.source.index_column] == phone][
            self.source.feature_columns
        ].values.flatten()
        phone2_df = phones_df[phones_df[self.source.index_column] == other_phone][
            self.source.feature_columns
        ].values.flatten()
        if distance_weights is None:
            return distance_fn(phone1_df, phone2_df)
        else:
            distance_weights = np.array(distance_weights)
            distance_weights = (
                distance_weights / distance_weights.sum() * len(distance_weights)
            )
            return distance_fn(
                phone1_df * distance_weights, phone2_df * distance_weights
            )

    def get_closest_by_vector(
        self,
        vector: List[float],
        distance_fn: Callable[
            [Iterable[float], Iterable[float]], float
        ] = distance.euclidean,
    ) -> List[Tuple[float, object]]:
        """Given a vector, find the phone that is closest to the vector

        Args:
            vector: The vector we're looking for the closest phones to.
            distance_fn: The function that will be used to calculate the distance between the vector and phones.

        Returns:
            A list of tuples, where each tuple contains a distance and a phone.
        """
        phones_df = (
            self.data.groupby(
                [c for c in self.columns if c != self.source.allophone_column]
            )
            .mean()
            .reset_index()
        )
        phones_df.drop_duplicates(inplace=True)
        smallest_dist = float("inf")
        results = []
        for _, row in phones_df.iterrows():
            dist = distance_fn(vector, row[self.source.feature_columns].values)
            if dist <= smallest_dist:
                if dist < smallest_dist:
                    results = []
                    smallest_dist = dist
                results.append((dist, self._row_to_phone(row)))
        return results

    def get_closest_by_phone(
        self,
        phone: List[float],
        distance_fn: Callable[
            [Iterable[float], Iterable[float]], float
        ] = distance.euclidean,
    ) -> List[Tuple[float, object]]:
        """Given a phone, return the closest phone in the collection

        Args:
            phone: The phone to find the closest phone to.
            distance_fn: The function that will be used to measure the distance between phones.

        Returns:
            A list of tuples, where each tuple contains a distance and a phone.
        """
        return self.get_closest_by_vector(phone.vector, distance_fn)

    def get_closest(
        self,
        phone: str,
        src_language: str,
        tgt_language: str,
        return_allophones: bool = False,
        distance_fn: Callable[
            [Iterable[float], Iterable[float]], float
        ] = distance.euclidean,
        distance_weights=None,
        allow_allophones=True,
        return_all=False,
    ) -> Union[
        List[Tuple[float, str]], Tuple[List[Tuple[float, str]], List[Tuple[float, str]]]
    ]:
        """Given a phone, a source language, a target language, a distance function, and a distance weight,
        `get_closest` returns the closest phone in the target language.
        It also returns all the allophones of the source phone in the target language.
        It also returns the distance between the source phone and the closest phone.

        Example:
        Let's say we want to find the closest phone to the phone `ð` in the language `English` in `German`.
        ```py
        pc = PhoneCollection()
        pc.get_closest("ð", "eng", "deu")
        ```
        > ``[(2.8284271247461903, 'z'), (2.8284271247461903, 'ʒ')]``

        Args:
            phone: The phone to be mapped.
            src_language: The language of the phone that you want to find the closest one to.
            tgt_language: The language of the target phone.
            return_allophones: If True, return a tuple of ``(closest_phones, allophones)``
            distance_fn: The distance function to use.
            distance_weights:
                If None, the distance weights are set to 1/n, where n is the number of features.
                Otherwise, the weights are normalised and then used for the distance calculations.
            allow_allophones: If True, then if the phone is not found in the inventory, search for a phone the given ``phone`` is an allophone of.
            return_all: If True, return all phones and their distances, not just the closest ones.

        Returns:
            Returns a list of ``(distance, phone)`` for the closests phones or for all phones if ``return_all`` is True.
            If ``allow_allophones`` is True, returns a Tuple of lists with the first entry being the closests phones and the second being allophones.
        """
        assert self.source.language_column is not None
        phone = unicodedata.normalize("NFC", phone)
        src_phone_df = self._get_phone_inventory(src_language)
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
        tgt_phone_df = self._get_phone_inventory(tgt_language)
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
                distance_weights = (
                    distance_weights / distance_weights.sum() * len(distance_weights)
                )
            dist = distance_fn(
                src_phone_vec * distance_weights, tgt_phone_vec * distance_weights
            )
            for _, row in rows.iterrows():
                if phone in row[self.source.allophone_column].split():
                    p = self._row_to_phone(row)
                    p.index = phone
                    allophones.append((dist, p))
                    break
            if dist <= smallest_dist or return_all:
                if dist < smallest_dist:
                    smallest_dist = dist
                    if not return_all:
                        results = []
                results.append((dist, self._row_to_phone(row)))

        if return_allophones:
            return sorted(results), allophones
        else:
            return sorted(results)

    def _row_to_phone(self, row):
        idx = row[self.source.index_column]
        features = {f: row[f] for f in self.source.feature_columns if row[f] != "N"}
        language = None
        if self.source.language_column in row:
            language = row[self.source.language_column]
        allophones = None
        if self.source.allophone_column in row:
            allophones = row[self.source.allophone_column]
        if len(features) > 0:
            try:
                _master = self._master
            except AttributeError:
                _master = self
            return Phone(idx, features, language, allophones, _master)
        return None
