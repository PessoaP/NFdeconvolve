# This file is a "forwarding" module. 
# It allows the nf_class module to remain usable in this folder, even though the actual library (NFdeconvolute.py) is in a different folder.
# We moved the nf_class code to the parent directory and renamed it to NFdeconvolute.py for better organization. However, some scripts in this folder still expect to import "nf_class".

import sys  
import os   

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from NFdeconvolute import *
