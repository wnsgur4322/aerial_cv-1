#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

//1st arg is path, 2nd arg is filename
int main(int argc, char ** argv )
{
	if (argc != 2)
	{
		std::cout<<"need image"<<std::endl;
	}
// Mat object doesnt require allocation of memory contains matrix header
//Use of the copy constructor i.e. Mat B(A) will only copy the headers and pointer to the large matrix and not the 
//data itself. 
	Mat image; 
//imread function reads an image into a Mat object. first argument is file name. second argument is a flag 
//letting imread determine how colors are handled. -1 corresponds to unchanged 1 corresponds to color
	image = imread(argv[1],1);
// .data is a pointer to the data
	if(image.data == NULL)
	{
		print("image didnt load");
	}

}
