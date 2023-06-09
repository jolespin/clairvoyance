#!/usr/bin/env python
# =======
# Contact
# =======
# Producer: Josh L. Espinoza
# Contact: jespinoz@jcvi.org, jol.espinoz@gmail.com
# Google Scholar: https://scholar.google.com/citations?user=r9y1tTQAAAAJ&hl
# =======
# License BSD-3
# =======
# https://opensource.org/licenses/BSD-3-Clause
#
# Copyright 2018 Josh L. Espinoza
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# =======
# Version
# =======
__version__= "2023.6.9"
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
