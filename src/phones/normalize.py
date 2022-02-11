from unicodedata import normalize


def normalize_unicode(x):
    x = normalize("NFC", x)
    x = x.replace("ɝ", "ɜ˞")
    x = x.replace("ɚ", "ə˞")
    return x
