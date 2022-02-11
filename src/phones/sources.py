"""
Sources for phone features.

## ``sources.PHOIBLE``

Source for phone inventories from [phoible.org](https://phoible.org).
Use the following citation:

```
@article{moran2014phoible,
title={PHOIBLE online},
author={Moran, Steven and McCloy, Daniel and Wright, Richard},
year={2014},
publisher={Max Planck Institute for Evolutionary Anthropology}
}
```

## ``sources.PANPHON``

Source for phone inventories from [panphon](https://github.com/dmort27/panphon).
Use the following citation:

```
@inproceedings{mortensen2016panphon,
title={Panphon: A resource for mapping IPA segments to articulatory feature vectors},
author={Mortensen, David R and Littell, Patrick and Bharadwaj, Akash and Goyal, Kartik and Dyer, Chris and Levin, Lori},
booktitle={Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers},
pages={3475--3484},
year={2016}
}
```
"""

from dataclasses import dataclass
from typing import List


@dataclass
class PhoneSource:
    """
    The PhoneSource class is a dataclass that stores the information about a phone source.

    A phone source is a csv containing phone definitions and their linguistic features

    Attributes:
        urls: a list of urls to the csv files
        index_column: the name of the column that contains the index (ipa character(s)) of the phone
        feature_columns: a list of the names of the columns that contain the features of the phone
        language_column: the name of the column that contains the iso code of the language
        dialect_column: the name of the column that contains the dialect
    """

    urls: List[str]
    index_column: str
    feature_columns: List[str]
    language_column: str = None
    allophone_column: str = None
    dialect_column: str = None


PHOIBLE = PhoneSource(
    ["https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv"],
    "Phoneme",
    [
        "tone",
        "stress",
        "syllabic",
        "short",
        "long",
        "consonantal",
        "sonorant",
        "continuant",
        "delayedRelease",
        "approximant",
        "tap",
        "trill",
        "nasal",
        "lateral",
        "labial",
        "round",
        "labiodental",
        "coronal",
        "anterior",
        "distributed",
        "strident",
        "dorsal",
        "high",
        "low",
        "front",
        "back",
        "tense",
        "retractedTongueRoot",
        "advancedTongueRoot",
        "periodicGlottalSource",
        "epilaryngealSource",
        "spreadGlottis",
        "constrictedGlottis",
        "fortis",
        "raisedLarynxEjective",
        "loweredLarynxImplosive",
        "click",
    ],
    "ISO6393",
    "Allophones",
    "SpecificDialect",
)

PANPHON = PhoneSource(
    [
        "https://raw.githubusercontent.com/dmort27/panphon/master/panphon/data/ipa_all.csv",
        "https://raw.githubusercontent.com/dmort27/panphon/master/panphon/data/ipa_bases.csv",
    ],
    "ipa",
    [
        "syl",
        "son",
        "cons",
        "cont",
        "delrel",
        "lat",
        "nas",
        "strid",
        "voi",
        "sg",
        "cg",
        "ant",
        "cor",
        "distr",
        "lab",
        "hi",
        "lo",
        "back",
        "round",
        "velaric",
        "tense",
        "long",
        "hitone",
        "hireg",
    ],
)
