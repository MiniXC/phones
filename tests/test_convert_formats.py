from phones.convert import Converter

converter = Converter()


# The following tests are from
# https://github.com/rossellhayes/ipa/blob/main/tests/testthat/test-ipa.R
def test_convert_xsampa_ipa():
    assert "ˈnom.bɾe" == converter('"nom.b4e', "xsampa", "ipa", return_str=True)
    assert "nɔ̃bʁ" == converter("nO~bR",  "xsampa", "ipa", return_str=True)
    assert "ˌhɛˈloʊ" == converter('%hE"loU',  "xsampa", "ipa", return_str=True)
    assert "ˌfoʊ.naɪˈif" == converter("/%foU.naI\"if/",  "xsampa", "ipa", return_str=True)
    assert "wɜ˞ld" == converter("w3`ld",  "xsampa", "ipa", return_str=True)

def test_convert_ipa_xsampa():
    assert '"nom.b4e' == converter('ˈnom.bɾe', "ipa", "xsampa", return_str=True)
    assert "nO~bR" == converter("nɔ̃bʁ",  "ipa", "xsampa", return_str=True)
    assert '%hE"loU' == converter('ˌhɛˈloʊ',  "ipa", "xsampa", return_str=True)
    assert "%foU.naI\"if" == converter("ˌfoʊ.naɪˈif",  "ipa", "xsampa", return_str=True)
    assert "w3`ld" == converter("wɜ˞ld",  "ipa", "xsampa", return_str=True)

def test_convert_arpa_ipa():
    assert "ɑ ɹ p ʌ" == converter("AA R P AH", "arpabet", "ipa", return_str=True)
    assert "h ɛ l oʊ" == converter("HH EH L OW", "arpabet", "ipa", return_str=True)
    assert "w ɜ˞ l d" == converter("W ER L D", "arpabet", "ipa", return_str=True)

def test_convert_ipa_arpa():
    assert "AA R P AH" == converter("ɑɹpʌ", "ipa", "arpabet", return_str=True)
    assert "HH EH L OW" == converter("hɛloʊ", "ipa", "arpabet", return_str=True)
    assert "W ER L D" == converter("w ɜ˞ l d", "ipa", "arpabet", return_str=True)

def test_convert_arpa_xsampa():
    assert "A r\\ p V" == converter("AA R P AH", "arpabet", "xsampa", return_str=True)
    assert "h E l oU" == converter("HH EH L OW", "arpabet", "xsampa", return_str=True)
    assert "w 3` l d" == converter("W ER L D", "arpabet", "xsampa", return_str=True)

def test_convert_xsampa_arpa():
    assert "AA R P AH" == converter("A r\\ p V", "xsampa", "arpabet", return_str=True)
    assert "HH EH L OW" == converter("h E l oU","xsampa", "arpabet", return_str=True)
    assert "W ER L D" == converter("w 3` l d", "xsampa", "arpabet", return_str=True)

def test_convert_normalization():
    assert "W ER L D" == converter("w ɝ l d", "ipa", "arpabet", return_str=True)
    assert "W ER0 L D" == converter("w ɚ l d", "ipa", "arpabet", return_str=True)