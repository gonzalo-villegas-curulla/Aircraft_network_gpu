# conda activate cugraphenv

import numpy as np
import cugraph
# import cudf
import json
from pprint import pprint


thefile = 'data_20241029_103747.json'
thefile = 'data_20241029_111406.json'

with open(thefile) as f:

    content = f.read()
    content = content.replace("][",",")
    a = 3
    # d = json.load(f)



a = 2