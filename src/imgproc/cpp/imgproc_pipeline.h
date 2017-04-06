#ifndef IMGPROC_H
#define IMGPROC_H

#include <libgen.h>
#include <fstream>
#include <iostream>
#include <string>
#include <ctime>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include "imgproc_defs.h"
#include "image.h"

class ImgProcPipeline{

	private:
		// Image object
		Image current_image;
		std::ofstream metadata_file;
		std::ifstream image_file;
		std::string output_dir;
		ImgprocMode mode;
	public:
		// Will open up the initialize image_file ifstream object given input_file string
		ImgProcPipeline(int, char **);		
		
		// Preprocesses the image before image processing
		void preprocess(const cv::Mat &, cv::Mat &);

		std::pair<int, int> getRescaledDimensions(const cv::Mat &, int, int);

		// Detect circles in the image
		void find_circles(const cv::Mat &);

		// Invokes get_next_image and then initializes an image object
		void run_single_image(const cv::Mat &);
		
		// Will process all of the images in the input file
		// Utilizes grab_next_image() to grab the next image
		void run_all();
		
		// Will grab the next image from the input file
		bool get_next_image(); 	

};

#endif
