This tool will run the image processor and summarize the results in an HTML file. This script 
will take care of downloading the images from Google Cloud Storage and building the image processor.
This tool assumes access to the `northamericaneclipseimages` Google Cloud Project

To run the tool:

```bash
$ ./test DIR
```
 `DIR` is the directory to use for image/data storage. The following will be saved into `DIR`:
 
 - A clone of the GCS bucket `eclipse_image_ground_truth_dataset_renamed`
 - A directory called `output` that will contain:
   - All the processed images with sun/moon circles overlayed from the `eclipse_image_ground_truth_dataset_renamed` bucket
   - A file called `metadata.txt` that will contain the processed image names along with output information from the image processor
   - A file called `output.html` that includes summarizes the image processor output info.

The `output.html` file will be automatically opened with Google Chrome
