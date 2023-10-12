import numpy as np
import tensorflow as tf
import re
import time

lines= open("Dataset\movie_lines.tsv", encoding= "utf-8",errors="ignore").read().split("\n")