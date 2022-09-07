from typing import Dict, List, Optional, Tuple, Union
from unicodedata import normalize
import numpy as np


class PhoneFeature:
    def __init__(self, feature: str, value: float) -> None:
        """
        Create a new instance of a ``PhoneFeature`` with the feature and value provided.

        Args:
            feature: The feature to be evaluated.
            value: The value of the feature.
        """
        self.feature = feature.lower()
        self.value = value


class Phone:
    def __init__(
        self,
        index: str,
        features: Dict[str, Union[int, str]],
        language_code: Optional[str] = None,
        allophones: Optional[List[str]] = None,
        collection: Optional[object] = None,
    ) -> None:
        """Create a new `Phone` object.

        Args:
            index: The index of the phone in the phone set.
            features: A dictionary of features. When not provided with numerical values, `-` will be replaced with `-1` and `+` with `1`.
            language_code: The language code of the language that the phoneme belongs to.
            allophones: A list of allophones for the phoneme.
            collection: The ``PhoneCollection`` the phone is contained in.

        Example:
            The ``Phone`` class supports arithmetic operations.
            ```py
            pc = PhoneCollection()
            pc.phones("i").val + pc.phones("u").val
            ```
            > ``[(0.7071067811865476, iu (adn)),(0.7071067811865476, iu (bhg)),...]``

            And data augmentation.
            ```py
            pc = PhoneCollection()
            z = pc.phones("z").val
            z_noise = pc.phones("z").val.noise(.05, random_state=42)
            z.vector - z_noise.vector
            ```
            > ``array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,``
            > ``        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,``
            > ``        0.,  0.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])``

            If the phones vector has been altered, we can also find its closest existing phone(s).
            ```py
            pc = PhoneCollection()
            z_noise = pc.phones("z").val.noise(.05, random_state=42).closest()
            ```
            > ``[(0.0, z̤ (xho))]``

            \... filtered by language(s)
            ```py
            pc = PhoneCollection()
            z_noise = pc.phones("z").val.noise(.05, random_state=42).langs("eng").closest()
            ```
            > ``[(2.0, z (eng))]``
        """
        self.index = normalize("NFC", index)
        self.feature_names = [f.lower() for f in sorted(features.keys())]
        features = {k.lower(): v for k, v in features.items()}
        self.features = features
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
        self.collection = collection

    def get_feature_vector(self, features: List[str]) -> np.ndarray:
        """
        Get the feature vector of the phone for the features provided. 

        Returns:
            A numpy array of the feature vector.
        """
        return np.array([self.features[k] for k in self.feature_names if k in features])

    @staticmethod
    def _normalize(v) -> List[float]:
        v[v > 1] = -1
        v[v < -1] = 1
        return v.round()

    def __repr__(self) -> str:
        result = self.index
        if self.language_code is not None:
            result += f" ({self.language_code})"
        return result

    def __eq__(self, __o: object) -> bool:
        return np.allclose(self.vector, __o.vector)

    def __add__(self, __o: object) -> object:
        assert self.collection is not None
        vector = self.vector
        if isinstance(__o, PhoneFeature):
            vector[self.feature_names.index(__o.feature)] += __o.value
        else:
            assert self.collection.lang_filter == __o.collection.lang_filter
            vector += __o.vector
        return self.collection.get_closest_by_vector(np.clip(vector, -1, 1))

    def __sub__(self, __o: object) -> object:
        assert self.collection is not None
        vector = self.vector
        if isinstance(__o, PhoneFeature):
            vector[self.feature_names.index(__o.feature)] -= __o.value
        else:
            assert self.collection.lang_filter == __o.collection.lang_filter
            vector -= __o.vector
        return self.collection.get_closest_by_vector(np.clip(vector, -1, 1))

    def __lt__(self, __o: object) -> bool:
        return self.index < __o.index

    def langs(self, langs: str) -> object:
        """
        The langs function takes a string or list of languages and returns the phone with the languages
        filter applied.

        Args:
            langs: A list of language codes or single language code.

        Returns:
            A new instance of the class.
        """
        self.collection = self.collection.langs(langs)
        return self

    def noise(
        self,
        p: float = 0.005,
        abs_max_change: float = 2,
        return_close=False,
        random_state: int = None,
    ) -> Union[List[Tuple[float, object]], object]:
        """
        Given a phone, it will return a new phone with a random vector that is close to the original phone.

        Args:
            p: The element-wise probability of a change in a phone vector.
            abs_max_change: The maximum absolute value an element of the phone vector can change.
            random_state: Seed used for the random numbers used.

        Returns:
            The phone object with a noised feature vector.
        """
        if random_state is not None:
            np.random.seed(random_state)
        probs = np.random.uniform(size=len(self.vector))
        self.vector[probs <= p] += np.random.uniform(
            -abs_max_change, abs_max_change, size=len(self.vector)
        )[probs <= p]
        self.vector = Phone._normalize(self.vector)
        return self

    def closest(self) -> List[Tuple[float, object]]:
        """
        Given the current phone's vector, return the closest phone(s) in the collection and their distances.

        Returns:
            A list of distance,phone tuples.
        """
        return self.collection.get_closest_by_vector(self.vector)
