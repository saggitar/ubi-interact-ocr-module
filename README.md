# ubii-interact-ocr-module
Tesseract OCR for Ubi Interact

## Requirements
* [tesseract] OCR Engine with appropriate language packs (https://tesseract-ocr.github.io/) _(>= 4.0.0)_
* [``tesserocr``](https://github.com/sirfz/tesserocr)
* opencv (some of the modules try to improve OCR performance with additional preprocessing, in the future this might become an optional requirement)

   > :warning: On windows ``tesserocr`` can't be installed from PyPi. Windows builds for tesserocr are available for [some python versions](https://github.com/simonflueckiger/tesserocr-windows_build/releases). Only python version __>=3.7 < 3.8__ is supported by both ``tesserocr`` and the ``ubi-interact-python`` node on Windows. Tesseract 5.+ should be compatible with Tesseract 4.0.0 (which is used for the Windows builds).
   
