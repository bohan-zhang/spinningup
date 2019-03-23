import glob
import os
import sys
import shutil

def exitWithUsage():
	print(' ')
	print('Usage:')
	print('   python filter_data_dir.py <data_dir> <output_folder> <size_limit_in_bytes>')
	print(' ')
	sys.exit()

if (len(sys.argv) < 4):
	exitWithUsage()

input_folder = sys.argv[1]
output_folder = sys.argv[2]
size_lim = int(sys.argv[3])


for filename in glob.glob(os.path.join(input_folder,'**/*'), recursive=True):
  file_size = os.stat(filename).st_size

  if os.path.isfile(filename) and file_size < size_lim:
    newfilename = '/'.join([output_folder] + filename.split('/')[1:])

    os.makedirs('/'.join(newfilename.split('/')[:-1]),exist_ok=True)
    shutil.copy2(filename, newfilename)
    print('copied {} to {}'.format(filename, newfilename))