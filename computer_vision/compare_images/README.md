# Image comparison

Image comparison using the Structural Similarity Index Measure (SSIM). SSIM has values in rage from -1 to 1 where 1 indicates that both images are identical. 

The script returns SSIM value converted into a percentage. If the optional flag --show is passed then the differences in both images will be highlighted and displayed (work in progress).

## Usage

```
usage: compare_images.py [-h] -a IMAGE_A -b IMAGE_B [--show]

optional arguments:
  -h, --help            show this help message and exit
  -a IMAGE_A, --image_a IMAGE_A
                        path to Image A
  -b IMAGE_B, --image_b IMAGE_B
                        path to Image B
  --show                display both images and the differences between them
```

## References

[[1](http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf)] 
Wang, Z., Bovik, A.C., Sheikh, H.R. and Simoncelli, E.P., 2004. Image quality assessment: from error visibility to structural similarity. 
IEEE transactions on image processing, 13(4), pp.600-612. 

[[2](http://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/)] pyimagesearch.com


