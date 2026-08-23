## Devanagari Akshara Segmentation

*A computer-vision pipeline for segmenting Devanagari/Sanskrit word images into individual aksharas (visual character clusters) using ground-truth text as guidance.*

## Pipeline Structure :-

The core idea is:

## Pipeline

```mermaid
flowchart TD
    A[Image + Ground Truth Text] --> B[Binarization]
    B --> C[Shirorekha Removal]
    C --> D[Noise Removal]
    D --> E[Ink-Run Detection]
    E --> F[Ground Truth Akshara Count]
    F --> G[Boundary Reconciliation]
    G --> H[Character / Akshara Crops]
    H --> I[PNG Crops]
    H --> J[JSON Manifest]
    H --> K[Debug Visualization]
```






# Project Structure

A typical input/output structure is:

~~~project/
│
├── Character Segmentation
│
├── input/
│   ├── line_0.jpg
│   └── line_0.txt
│   
│    
│
└── output/
    ├── line_0_w0_c0_<akshara>.png
    ├── line_0_segmentation_debug.png
    └── line_0_manifest.json
~~~

*Each image should have a matching .txt file containing its ground-truth text.*


Important Limitation

The segmentation is ground-truth guided.

The ground truth provides the expected akshara count and labels. Image evidence is then used to determine where the boundaries should be placed.

Therefore, this implementation should not be interpreted as a completely text-independent character segmentation system.

The python documentation recommends checking the generated:

*_segmentation_debug.png

on a sample of images and adjusting SegmentConfig parameters when necessary.


# Input:-

![input image](line_0.jpg)

And the file got after going through this pipeline *Debug_file* and character segmentation i got using the pipeline
![output_image](debugimage.png)



& 

![segmented_मि](line_20_w0_c0_मि.png)

![segmented_थु](line_20_w0_c1_थु.png)

![segmented_नं](line_20_w0_c2_नं.png)





