import os
import sys
import random

import json
import numpy as np
import pandas as pd

import macro as mc
import pymatgen as mg

def load_file():
	if os.path.isfile(mc._SOURCE_DATA)==0:
		print('error: no source domain data')
		sys.exit(1)
	elif os.path.isfile(mc._TARGET_DATA)==0:
		print('error: no target domain data')
		sys.exit(1)

	source_df = pd.read_csv(mc._SOURCE_DATA,delimiter=',',engine="python",header=None)
	target_df = pd.read_csv(mc._TARGET_DATA,delimiter=',',engine="python",header=None)

	return source_df, target_df
