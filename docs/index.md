# phones

**phones** is a python library for the easy handling of phones in the [International Phonetic Alphabet](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet).
These IPA phones can be useful because they can describe how words are pronounced in most languages.

## Feature Overview
- Extract numeric feature vectors from phones.
- Map phones from one language to another by finding the closest phones.
- Convert between [ARPABET](https://en.wikipedia.org/wiki/ARPABET), [X-SAMPA/SAMPA](https://en.wikipedia.org/wiki/X-SAMPA) and [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) notation.
- Compute phone distances.
- Do phone arithmetic on a phone and phone-feature level.
- Visualise phones using and their distances when installing ``phones[plots]``.

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

    This is the `0.0.1` release of this library, and things might be unstable.

    Please report issues to [https://github.com/MiniXC/phones]()