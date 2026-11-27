import sys
from PIL import Image
# crop.py <png> <out> <x0> <y0> <x1> <y1>  (fractions of page) [scale]
im = Image.open(sys.argv[1]); w,h = im.size
x0,y0,x1,y1 = [float(v) for v in sys.argv[3:7]]
sc = float(sys.argv[7]) if len(sys.argv)>7 else 1.0
box = im.crop((int(w*x0),int(h*y0),int(w*x1),int(h*y1)))
if sc!=1.0: box = box.resize((int(box.width*sc),int(box.height*sc)), Image.LANCZOS)
box.save(sys.argv[2]); print(box.size)
