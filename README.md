### MOVIE GENRE PREDICTION FROM SUBTITLE USING DEEP LEARNING
#### Master Thesis
## Paper
https://acikbilim.yok.gov.tr/bitstream/handle/20.500.12812/265957/yokAcikBilim_10307324.pdf?sequence=-1&isAllowed=y

## Dataset
I have used OpenSubtitles.org English movide dataset from https://opus.nlpl.eu/OpenSubtitles-v2018.php

## Applications

 1. build-dataset.py; This script uses IMDB to find genres of subtitle and creates id_match.csv.
 2.  build-pickle.py; This scripts create pickle database from dataset.csv
 3. run-cross-validation.py; This scripts load builded pickle files and run training and cross validation and creates result.txt