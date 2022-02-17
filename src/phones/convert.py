"""
This module allows to convert between the "ipa", "xsampa" and "arpabet" formats.
The code is adapted from the [phonecodes package](https://github.com/jhasegaw/phonecodes) by Mark Hasegawa-Johnson.

Examples:
    A converter object can be used.
    ```py
    from phone.convert import Converter
    converter = Converter()
    converter("wɜ˞ld", "ipa", "arpabet")
    ```
    > ``['W', 'ER', 'L', 'D']``

    You can also list all possible formats.
    ```py
    converter.formats
    ```
"""
from typing import List, Optional
import re

from .phonecodes.src import phonecodes
from phones.normalize import normalize_unicode

class Converter:
    def __init__(self) -> None:
        self.phonecodes = {
            ("arpabet", "ipa"): (phonecodes.arpabet2ipa, False),
            ("ipa", "arpabet"): (phonecodes.ipa2arpabet, False),
            ("ipa", "callhome"): (phonecodes.ipa2callhome, ["arz", "cmn", "spa"]),
            ("callhome", "ipa"): (phonecodes.callhome2ipa, ["arz", "cmn", "spa"]),
            ("ipa", "disc"): (phonecodes.ipa2disc, False),
            ("disc", "ipa"): (phonecodes.disc2ipa, ["nld", "eng", "deu"]),
            ("ipa", "xsampa"): (phonecodes.ipa2xsampa, False),
            ("xsampa", "ipa"): (phonecodes.xsampa2ipa, False),
            ("arpabet", "xsampa"): (self.arpabet2xsampa, False),
            ("xsampa", "arpabet"): (self.xsampa2arpabet, False),
        }

    def xsampa2arpabet(self, x, lang):
        if not lang:
            lang = None
        return self(
            self(
                x,
                "xsampa",
                "ipa",
                lang,
                True
            ),
            "ipa",
            "arpabet",
            lang, 
            True
        )

    def arpabet2xsampa(self, x, lang):
        if not lang:
            lang = None
        return self(
            self(
                x,
                "arpabet",
                "ipa",
                lang,
                True
            ),
            "ipa",
            "xsampa",
            lang,
            True
        )

    def __call__(
        self, x, _from: str, _to: str = "ipa", lang: Optional[str] = None, return_str = False,
    ) -> List[str]:
        """
        It takes a string, and replaces all the symbols of the ``_from`` format to the ``_to`` format.

        Args:
            x: the string to be converted
            _from: the name of the format to convert from
            _to: the name of the format to convert to

        Returns:
            The converted string.
        """
        func, langs = self.phonecodes[(_from, _to)]
        assert not (not langs and lang is not None)
        result = func(normalize_unicode(x), langs)
        result = [normalize_unicode(r) for r in result]
        if max([len(r) for r in result]) > 1:
            result_new = []
            for r in result:
                result_new += [r, ' ']
            result = result_new[:-1]
        if return_str:
            result = "".join(result)
            result = re.sub(r"\s+", " ", result).strip()
        else:
            result = [r for r in result if len(r.replace(" ", "")) > 0]
        return result

    @property
    def formats(self) -> List[str]:
        return list(set([x[0] for x in self.phonecodes.keys()]))
