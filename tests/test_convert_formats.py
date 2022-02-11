from lib2to3.pytree import convert
from phones.convert import converter


# The following tests are from
# https://github.com/rossellhayes/ipa/blob/main/tests/testthat/test-ipa.R
def test_convert_xsampa_ipa():
    assert "ˈnom.bɾe" == converter.convert('"nom.b4e', "xsampa", "ipa")
    assert "nɔ̃bʁ" == converter.convert("nO~bR",  "xsampa", "ipa")
    assert "ˌhɛˈloʊ" == converter.convert('%hE"loU',  "xsampa", "ipa")
    assert "ˌfoʊ.naɪˈif" == converter.convert("/%foU.naI\"if/",  "xsampa", "ipa")
    assert "wɜ˞ld" == converter.convert("w3`ld",  "xsampa", "ipa")

def test_convert_ipa_xsampa():
    assert '"nom.b4e' == converter.convert('ˈnom.bɾe', "ipa", "xsampa")
    assert "nO~bR" == converter.convert("nɔ̃bʁ",  "ipa", "xsampa")
    assert '%hE"loU' == converter.convert('ˌhɛˈloʊ',  "ipa", "xsampa")
    assert "%foU.naI\"if" == converter.convert("ˌfoʊ.naɪˈif",  "ipa", "xsampa")
    assert "w3`ld" == converter.convert("wɜ˞ld",  "ipa", "xsampa")

def test_convert_arpa_ipa():
    assert "ɑ ɹ p ʌ" == converter.convert("AA R P AH", "arpabet", "ipa")
    assert "h ɛ l oʊ" == converter.convert("HH EH L OW", "arpabet", "ipa")
    assert "w ɜ˞ l d" == converter.convert("W ER L D", "arpabet", "ipa")

def test_convert_ipa_arpa():
    assert "AA R P AH" == converter.convert("ɑɹpʌ", "ipa", "arpabet")
    assert "HH EH L OW" == converter.convert("hɛloʊ", "ipa", "arpabet")
    assert "W ER L D" == converter.convert("w ɜ˞ l d", "ipa", "arpabet")

def test_convert_arpa_xsampa():
    assert "A r\\ p V" == converter.convert("AA R P AH", "arpabet", "xsampa")
    assert "h E l oU" == converter.convert("HH EH L OW", "arpabet", "xsampa")
    assert "w 3` l d" == converter.convert("W ER L D", "arpabet", "xsampa")

def test_convert_xsampa_arpa():
    assert "AA R P AH" == converter.convert("A r\\ p V", "xsampa", "arpabet")
    assert "HH EH L OW" == converter.convert("h E l oU","xsampa", "arpabet")
    assert "W ER L D" == converter.convert("w 3` l d", "xsampa", "arpabet")

def test_convert_normalization():
    assert "W ER L D" == converter.convert("w ɝ l d", "ipa", "arpabet")
    assert "W ER0 L D" == converter.convert("w ɚ l d", "ipa", "arpabet")