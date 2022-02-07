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
import pandas as pd
import re
import pkg_resources


class Converter:
    def __init__(self) -> None:
        stream = pkg_resources.resource_stream(__name__, "data/phonemes.csv")
        self.df = pd.read_csv(stream)

    def convert(self, x, _from="xsampa", _to="ipa") -> str:
        """
        It takes a string, and replaces all the symbols of the ``_from`` format to the ``_to`` format.
        
        Args:
            x: the string to be converted
            _from: the name of the format to convert from
            _to: the name of the format to convert to
        
        Returns:
            The converted string.
        """
        df = self.df.dropna(subset=[_from, _to]).copy()
        df["empty"] = df[_to].apply(lambda x: len(x) == 0)
        df["length"] = df[_from].apply(lambda x: len(x))
        df = df.sort_values(["empty", "length", _from, _to], ascending=False)
        df = df.drop(columns=["empty", "length"])
        df[_from] = df[_from].apply(lambda x: f"{x}(?![^\\(]*[\\)])")
        df[_to] = df[_to].apply(lambda x: x.replace("\.", "."))
        df[_to] = df[_to].apply(lambda x: f"({x})")
        df = df.reset_index(drop=True)
        if _from == "arpabet":
            df.loc[len(df)] = ["", "", " "]
            df.loc[-1] = [" ", " ", "\\d"]
            df.index = df.index + 1
            df = df.sort_index()
        df.loc[len(df)] = ["", "", ""]
        df.iloc[-1][_from] = "\\(|\\)"
        x = f" {x} "
        x = x.replace(
            "ɜ˞", "ɝ"
        )  # necessary because this seems to happen automatically in R
        for _, row in df.iterrows():
            x = re.sub(row[_from], row[_to], x)
        return x.strip()


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
