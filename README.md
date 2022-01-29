# ubii-processing-module-ocr
Tesseract OCR for Ubi Interact

## Requirements
* [tesseract](https://tesseract-ocr.github.io/) OCR Engine with appropriate language packs _(>= 4.0.0)_
* opencv (some of the modules try to improve OCR performance with additional preprocessing, in the future this might become an optional requirement)
* numpy
* [``tesserocr``](https://github.com/sirfz/tesserocr)

   > :warning: On windows ``tesserocr`` can't be installed from PyPi. Windows builds for tesserocr are available for [some python versions](https://github.com/simonflueckiger/tesserocr-windows_build/releases). Only python version __>=3.7 < 3.8__ is supported by both ``tesserocr`` and the ``ubi-interact-python`` node on Windows. Tesseract 5.+ should be compatible with Tesseract 4.0.0 (which is used for the Windows builds).
   
## Known Issues
* The processing modules which use opencv seem to sometimes randomly crash under Windows
* Processing Frequency depends on the module implementation. Typically ``TesseractOCR_EAST`` allows for a slightly higher processing frequency and is run at __10fps__ while the ``TesseractOCR_PURE`` module is run at __5fps__.
