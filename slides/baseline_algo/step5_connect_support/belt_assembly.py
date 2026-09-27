"""Compatibility import; current assembly implementation lives in Step6."""
import sys
from step6_connect_support import belt_assembly as implementation
sys.modules[__name__] = implementation
