# Using InstanSeg in QuPath

InstanSeg has its own extension for [QuPath](https://qupath.github.io/), so you can segment
images without writing any code. Because the whole InstanSeg model, including postprocessing,
compiles to TorchScript, the extension runs it directly inside QuPath without Python.

InstanSeg is included in recent [QuPath releases](https://github.com/qupath/qupath/releases/).
The source code of the extension is in the
[qupath-extension-instanseg](https://github.com/qupath/qupath-extension-instanseg) repository.

GeoJSON files saved by InstanSeg in Python (see {doc}`inference`) can also be imported into
QuPath.
