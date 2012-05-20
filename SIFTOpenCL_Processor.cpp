/*!
 * \file SIFTOpenCL_Processor.cpp
 * \brief
 * \author mpruchniak
 * \date 2012-04-27
 */
 
 
#include <memory>
#include <string>

#include "SIFTOpenCL_Processor.hpp"
#include "Logger.hpp"

namespace Processors {
namespace SIFTOpenCL {

SIFTOpenCL_Processor::SIFTOpenCL_Processor(const std::string & name) : Base::Component(name)
{
	LOG(LTRACE) << "Hello SIFTOpenCL_Processor\n";

	siftGPU = new SiftGPU();


}

SIFTOpenCL_Processor::~SIFTOpenCL_Processor()
{
	LOG(LTRACE) << "Good bye SIFTOpenCL_Processor\n";
}

bool SIFTOpenCL_Processor::onInit()
{
	LOG(LTRACE) << "SIFTOpenCL_Processor::initialize\n";

	h_onNewImage.setup(this, &SIFTOpenCL_Processor::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	registerStream("in_img", &in_img);

	newImage = registerEvent("newImage");

	registerStream("out_img", &out_img);

	return true;
}

bool SIFTOpenCL_Processor::onFinish()
{
	LOG(LTRACE) << "SIFTOpenCL_Processor::finish\n";

	return true;
}

bool SIFTOpenCL_Processor::onStep()
{
	LOG(LTRACE) << "SIFTOpenCL_Processor::step\n";
	return true;
}

bool SIFTOpenCL_Processor::onStop()
{
	return true;
}

bool SIFTOpenCL_Processor::onStart()
{
	return true;
}

void SIFTOpenCL_Processor::onNewImage()
{
	LOG(LTRACE) << "SIFTOpenCL_Processor::onNewImage\n";
	try {
		cv::Mat img = in_img.read();
		IplImage ipl = img;
		cout << "SIFTOpenCL_Processor::onNewImage " << endl;
		
		
		clock_t start, finish;
		double duration = 0;
		start = clock();
		
			int numberDesc = siftGPU->DoSift(&ipl);
			features = siftGPU->feat;
			
		finish = clock();
		duration = (double)(finish - start) / CLOCKS_PER_SEC;
		cout << "PROCESSING TIME GPU: " << endl;
		cout << duration << endl;
		
		cout << "SIFTOpenCL_Processor - number desc " << numberDesc << endl;
		
		out_img.write(img);
		newImage->raise();
	} catch (...) {
		LOG(LERROR) << "SIFTOpenCL_Processor::onNewImage failed\n";
	}
}

}//: namespace SIFTOpenCL
}//: namespace Processors
