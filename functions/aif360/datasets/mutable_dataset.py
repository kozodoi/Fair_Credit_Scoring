from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from aif360.datasets import StructuredDataset


class MutableDataset(StructuredDataset):
    """Base class for mutable datasets.

    A MutableDataset defines directions and limits for changing features. This
    primarily intended to allow for recourse analysis.
    """

    def __init__(self):
        """
        Args:
            **kwargs: StructuredDataset arguments.
        """

        super(MutableDataset, self).__init__(**kwargs)
