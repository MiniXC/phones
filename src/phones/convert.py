"""
This module allows to convert between the "ipa", "xsampa" and "arpabet" formats.
The code is ported from the [R ipa package](https://github.com/rossellhayes/ipa) to python.
Please use the following citation to cite their work:
```
@Manual{,
title = {ipa: convert between phonetic alphabets},
author = {Rossell Hayes and {Alexander}},
year = {2020},
note = {R package version 0.1.0},
url = {https://github.com/rossellhayes/ipa},
}
```

Examples:
    Either, a converter object can be used.
    ```py
    from phone.convert import converter
    converter.convert("wɜ˞ld", "ipa", "arpabet")
    ```
    > ``'W ER L D'``

    Alternatively, there are wrapper classes for each lexicon.
    ```py
    from phone.convert import Ipa, Arpabet, XSampa
    Ipa("wɜ˞ld").to_arpabet()
    ```
    > ``'W ER L D'``
"""
from typing import Optional
import pandas as pd
import re
import pkg_resources

from .phonecodes.src import phonecodes
from phones.normalize import normalize_unicode

def arpabet2xsampa(x, lang):
    return phonecodes.ipa2xsampa(phonecodes.arpabet2ipa(x, lang), lang)

def xsampa2arpabet(x, lang):
    return phonecodes.ipa2arpabet(phonecodes.xsampa2ipa(x, lang), lang)

_phonecodes = {
        ('arpabet','ipa'): (phonecodes.arpabet2ipa,False),
        ('ipa','arpabet'): (phonecodes.ipa2arpabet,False),
        ('ipa','callhome'): (phonecodes.ipa2callhome,['arz','cmn','spa']),
        ('callhome','ipa'): (phonecodes.callhome2ipa,['arz','cmn','spa']),
        ('ipa','disc'): (phonecodes.ipa2disc,False),
        ('disc','ipa'): (phonecodes.disc2ipa,['nld','eng']),
        ('ipa','xsampa'): (phonecodes.ipa2xsampa,False),
        ('xsampa','ipa'): (phonecodes.xsampa2ipa,False),
        ('arpabet','xsampa'): (arpabet2xsampa, False),
        ('xsampa','arpabet'): (xsampa2arpabet, False),
}

class Converter:
    def __init__(self) -> None:
        stream = pkg_resources.resource_stream(__name__, "data/phonemes.csv")
        self.df = pd.read_csv(stream)

    def convert(self, x, _from:str="xsampa", _to:str="ipa", lang:Optional[str]=None) -> str:
        """
        It takes a string, and replaces all the symbols of the ``_from`` format to the ``_to`` format.
        
        Args:
            x: the string to be converted
            _from: the name of the format to convert from
            _to: the name of the format to convert to
        
        Returns:
            The converted string.
        """
        func, langs = _phonecodes[(_from, _to)]
        assert not (not langs and lang is not None)
        result = func(normalize_unicode(x), langs)
        result = normalize_unicode(result)
        result = re.sub('\s+', ' ', result).strip()
        return result


converter = Converter()

class Ipa:
    def __init__(self, phone: str) -> None:
        self.x = phone

    def to_arpabet(self) -> object:
        global converter
        return Arpabet(converter.convert(self.x, "ipa", "arpabet"))

    def to_xsampa(self) -> object:
        global converter
        return XSampa(converter.convert(self.x, "ipa", "xsampa"))

    def __repr__(self) -> str:
        return self.x


class Arpabet:
    def __init__(self, phone: str) -> None:
        self.x = phone

    def to_ipa(self) -> object:
        global converter
        return Ipa(converter.convert(self.x, "arpabet", "ipa"))

    def to_xsampa(self) -> object:
        global converter
        return XSampa(converter.convert(self.x, "arpabet", "xsampa"))

    def __repr__(self) -> str:
        return self.x


class XSampa:
    def __init__(self, phone: str) -> None:
        self.x = phone

    def to_arpabet(self) -> object:
        global converter
        return Arpabet(converter.convert(self.x, "xsampa", "arpabet"))

    def to_ipa(self) -> object:
        global converter
        return Ipa(converter.convert(self.x, "xsampa", "ipa"))

    def __repr__(self) -> str:
        return self.x
