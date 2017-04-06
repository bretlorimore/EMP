#include <opencv2/highgui/highgui.hpp>

#include "image.h"


using std::endl;
using std::map;
using std::string;
using std::vector;

using namespace std;
using namespace cv;

Image::Image(){};

Image::Image(const cv::Mat &image, ImgprocMode mode, std::ofstream *metadata_file, const string &image_dest)
: original(image), mode(mode), metadata_file(metadata_file), dest(image_dest)
{
}

void Image::add_intermediate_image(const string &name, const Mat &image)
{
    this->intermediate_images[name] = image;
}

void Image::add_final_image(const Mat &image)
{
    this->processed = image;
}

bool Image::record()
{
    int key = 0;
    map<string, Mat>::iterator    img_it;
    map<string, double>::iterator time_it;
    vector<Vec3f>::iterator       circle_it;
    vector<string>::iterator      obs_it;      

    switch (this->mode)
    {
    case WINDOW:

        // Show original image
        imshow("original", this->original);

        // Show intermediate images
        for (img_it = this->intermediate_images.begin(); img_it != this->intermediate_images.end(); img_it++)
        {
            imshow(img_it->first, img_it->second);
        }

        // Show final image
        imshow("processed", this->processed);

        // Wait for user to press any key
        key = waitKey(0);
        destroyAllWindows();
        break;

    case BATCH:
        // Save processed image
        imwrite(this->dest, this->processed);

        // Save filename metadata
        *(this->metadata_file) << this->dest << METADATA_SEP;
       
        // Save circle metadata
        for (circle_it = this->circles.begin(); circle_it != this->circles.end(); circle_it++)
        {
            *(this->metadata_file) << "c(" << (*circle_it)[0] << "," << (*circle_it)[1] << "," << (*circle_it)[2] << ")" << METADATA_SEP;
        }

        // Save time metadata
        for (time_it = this->execution_times.begin(); time_it != this->execution_times.end(); time_it++)
        {
            *(this->metadata_file) << "t(\"" << time_it->first << "\"," << time_it->second << ")" << METADATA_SEP;
        }

        // Save observation metadata
        for (obs_it = this->observations.begin(); obs_it != this->observations.end(); obs_it++)
        {
            *(this->metadata_file) << "\"" << *obs_it << "\""  << METADATA_SEP;
        }

        // End line
        *(this->metadata_file) << endl;

        break;

    default:
        break;
    }

    return key == ESC_KEY;
}

void Image::add_circle(const Vec3f &circle)
{
    this->circles.push_back(circle);
}

void Image::add_circles(const vector<Vec3f> &new_circles)
{
    this->circles.reserve(this->circles.size() + std::distance(new_circles.begin(), new_circles.end()));
    this->circles.insert(this->circles.end(), new_circles.begin(), new_circles.end());
}

void Image::add_observation(const std::string &observation)
{
    this->observations.push_back(observation);
}

void Image::add_execution_time(const std::string &name, double num_secs)
{
    this->execution_times[name] = num_secs;
}
