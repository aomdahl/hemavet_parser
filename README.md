# hemavet_parser

This is a quick python script that uses the Google OpenVision API to parse the output of a HEMAVET 950FS machine.
The output captures printed results with reasonably high fidelity. However, handwritten notes on the paper may or may not be well represented.

## Usage
```
python3.9 hemavet_to_df.py --i <path_to_image_dir> -o <output_file_to_write_to>
```
Where `path_to_image_dir` is a directory containing all the image files (preferable .jpegs) that you'd like to parse, and `output_file_to_write_to` is the name of the table file this will be written to.

## Setting up google cloud service account
TBD

## Dependencies:
**Software**
```
python3.9
```
**Python package dependencies**
```
google
argparse
cv2
io
numpy
re
os
sys
pdb
```
