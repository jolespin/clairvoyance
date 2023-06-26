#!/usr/bin/env python
# =======
# Contact
# =======
# Producer: Josh L. Espinoza
# Contact: jespinoz@jcvi.org, jol.espinoz@gmail.com
# Google Scholar: https://scholar.google.com/citations?user=r9y1tTQAAAAJ&hl
# =======
# License:
# =======
#                     GNU AFFERO GENERAL PUBLIC LICENSE
#                        Version 3, 19 November 2007
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.
#
# =======
# Version
# =======
__version__= "2023.6.26"
__author__ = "Josh L. Espinoza"
__email__ = "jespinoz@jcvi.org, jol.espinoz@gmail.com"
__url__ = "https://github.com/jolespin/clairvoyance"
__cite__ = "https://github.com/jolespin/clairvoyance"
__license__ = "BSD-3"
__developmental__ = True

functions = { 
  "format_stratify",
  "format_weights",
  "format_cross_validation",
  "get_feature_importance_attribute",
  "get_balanced_class_subset",
  "recursive_feature_inclusion",
  "plot_scores_line","plot_weights_bar",
  "plot_weights_box",
  "plot_recursive_feature_selection",
  "plot_scores_comparison",
}
classes = {
  "ClairvoyanceBase","ClairvoyanceClassification","ClairvoyanceRegression","ClairvoyanceRecursive",

}
__all__ = sorted(functions | classes)
__doc__ = """

 _______        _______ _____  ______ _    _  _____  __   __ _______ __   _ _______ _______
 |       |      |_____|   |   |_____/  \  /  |     |   \_/   |_____| | \  | |       |______
 |_____  |_____ |     | __|__ |    \_   \/   |_____|    |    |     | |  \_| |_____  |______
                                                                                           

"""
# =======
from .clairvoyance import *
