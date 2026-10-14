"""HaSeen 3: word-level Arabic dialect deviation detection: MSA, Egyptian, Gulf (Saudi, Gulf-east,
Yemeni, Omani), Saudi, Levantine, Iraqi; Maghrebi and other dialects are named when they appear.

    import haseen
    haseen.detect("أبغى أساعدك، سوف أحوّلك للقسم المختص الحين", "egy")
"""
__version__ = "3.0.0"


def __getattr__(name):  # lazy: importing haseen.text must not load onnxruntime or torch
    if name in ("detect", "load", "Detector"):
        from . import api
        return getattr(api, name)
    raise AttributeError(name)
