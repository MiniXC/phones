# phones

**phones** is a python library for the easy handling of phones in the [International Phonetic Alphabet](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet).
These IPA phones can be useful because they can describe how words are pronounced in most languages.

## Feature Overview
- Extract numeric feature vectors from phones.
- Map phones from one language to another by finding the closest phones.
- Convert between [ARPABET](https://en.wikipedia.org/wiki/ARPABET), [X-SAMPA/SAMPA](https://en.wikipedia.org/wiki/X-SAMPA), [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet), [DISC](https://groups.linguistics.northwestern.edu/speech_comm_group/documents/CELEX/Phonetic%20codes%20for%20CELEX.pdf) and [CALLHOME](https://catalog.ldc.upenn.edu/LDC97L20) notation.
- Compute phone distances.
- Do phone arithmetic on a phone and phone-feature level.
- Visualise phones and their distances when installing ``phones[plots]``.

## Installation

For the core libary:
```bash
pip install phones
```

For plotting:
```bash
pip install phones[plots]
```

!!! note

    This is the `0.0.2` release of this library, and things might be unstable.

    Please report issues to [https://github.com/MiniXC/phones]()