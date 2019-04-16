# Fundus-image-preprocessing
Fundus image preprocessing. Uing Python3.5,OpenCV,Numpy.

The algorithm of fundus image preprocessing can be divided into 4 steps:
1.delete meaningless margin areas.
  Most fundus images contain some meaningless margin areas(for example black margin areas).
  We try to find the line between black margin areas and the region of retina, and delete these meaningless margin areas.

2.detect the circle of retina.
  Based on our domain knowledge, we know the retina region is a circle.
  Firstly we use HoughCircles detect the circle of retina. If not found, we suppose the center of the image is the center of circle.
  And get radius length based on the pixels distribution of the middle line.

3.crop the image based on detected retina circle in step 3.

4.add black margin.
  Afterwards we do image augmentation(random rotation and clip) afterwards. 
  To avoid deleting meaningful border areas, during training we add some black border areas to fundus images.
