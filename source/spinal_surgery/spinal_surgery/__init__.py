"""
Python module serving as a project/extension template.
"""

import os

ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))

# Keep the tracked Python asset configuration in this package while allowing the
# large published asset data to live on Blue storage outside the Git checkout.
ASSETS_DATA_DIR = os.path.abspath(
    os.environ.get("SONOGYM_ASSETS_DATA_DIR", os.path.join(ASSETS_EXT_DIR, "data"))
)

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
MODELS_DIR = os.path.abspath(os.environ.get("SONOGYM_MODELS_DIR", os.path.join(PROJECT_DIR, "models")))

# Define paths before task discovery imports the asset configuration modules.
from .lab import *
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *
