import os
import os
from PIL import Image

# Specify the directory path
directory_path = "Screenshots" 

# Text file name (replace with desired name)
output_file = "filenames.txt"

# Get list of filenames in the directory
filenames = os.listdir(directory_path)

# find scales:
widths = []
for filename in filenames:
    # determine scale
    img = Image.open(directory_path+"/" + filename) 
    # get width
    width = img.width 
    img.close()
    widths.append(width)

max_width = max(widths)
scales = [round(width/max_width, 2) for width in widths]


# Open the output file in append mode ('w')
with open(output_file, 'w') as text_file:
    # Write each filename to the text file
    for item in enumerate(filenames):
        index, filename = item
        text_file.write("\includegraphics[width=" + str(scales[index]) + "\linewidth,keepaspectratio]{Screenshots/" + filename + "}\n")

print(f"Filenames written to: {output_file}")
