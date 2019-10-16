#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>

typedef cv::cuda::GpuMat GpuMat;
typedef cv::Mat Mat;

#include <chrono>

#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include <cassert>

namespace py = boost::python;
namespace np = boost::numpy;

/* Convertion Between Python and C++ opencv */
np::ndarray toPython(const cv::Mat &frame );
const cv::Mat get_Mat(const np::ndarray &py_img);

np::ndarray toPython(const cv::Mat &frame){

  // np::ndarray array = np::from_data(frame.data)
  int height = frame.size().height;
  int width = frame.size().width ;
  int channels = frame.channels();

  assert(channels == 2);
  assert(frame.isContinuous())

  np::ndarray np_array = np::from_data(
    frame.data,
    np::dtype::get_builtin<float>(),
    py::make_tuple(height * width * channels),
    py::make_tuple(sizeof(float)),
    py::object()
  );
  return np_array;
}

const Mat get_Mat(const np::ndarray &py_img){
  const Py_intptr_t *shape = py_img.get_shape();
  const cv::Mat mat = cv::Mat(shape[0], shape[1], CV_8UC1, py_img.get_data());
  return mat;
}
/* End Convertion*/

class Flow {

// public method
public:
  void set_prev_frame(const Mat &img);
  Mat get_flow(const GpuMat &cur_img);
  Mat get_flow(const Mat &cur_img);
  np::ndarray get_py_flow(py::object py_img);

// private variables
private:
  GpuMat m_prev_img;
  GpuMat m_flow;
  Mat m_flow_mat;
  int m_width;
  int m_height;
  cv::Ptr<cv::cuda::FarnebackOpticalFlow> m_flow_calc;

// private methods
private:
  Mat resize(const Mat &img);
  void init(const Mat &img, int width=640, int height=360);

// constructor
public:
  Flow(const Mat &img, int width=640, int height=360){
    init(img, width, height);
  }
  Flow(py::object py_img){
    np::ndarray np_img = np::from_object(py_img);
    const Mat img = get_Mat(np_img);
    init(img);
  }
};

inline void Flow::set_prev_frame(const Mat &img){
  Mat resized_img = resize(img);
  m_prev_img.upload(resized_img);
}

void Flow::init(const Mat &img, int width, int height){
  assert (img.channels() == 1 && "Only Black & White Image");
  m_width = width;
  m_height = height;
  set_prev_frame(img);
  m_flow_calc = cv::cuda::FarnebackOpticalFlow::create();
}

Mat Flow::resize(const Mat &img){
  Mat resized_img;
  cv::resize(img, resized_img, cv::Size(m_width, m_height), 0, 0, cv::INTER_AREA);
  return resized_img;
}

Mat Flow::get_flow(const GpuMat &cur_img){
  m_flow_calc->calc(m_prev_img, cur_img, m_flow);
  Mat flow;
  m_flow.download(flow);
  m_prev_img = cur_img;
  return flow;
}

Mat Flow::get_flow(const Mat &cur_img){
  Mat resized_img = resize(cur_img);
  const GpuMat gpu_cur_img(resized_img);
  return get_flow(gpu_cur_img);
}

np::ndarray Flow::get_py_flow(py::object py_img){
  np::ndarray np_img = np::from_object(py_img);
  const Mat img = get_Mat(np_img);
  m_flow_mat = get_flow(img);
  return toPython(m_flow_mat);
}

BOOST_PYTHON_MODULE(cpp_flow){
  Py_Initialize();
  np::initialize();
  py::class_<Flow>("Flow", py::init<py::object>())
    .def("get_flow", &Flow::get_py_flow)
  ;
}

// int main(int argc, char** argv){
//   const Mat img1 = cv::imread("../videos/01/000001.jpg", cv::IMREAD_GRAYSCALE);
//   assert(!img1.empty());
//   Flow flow(img1);
//
//   const Mat img2 = cv::imread("../videos/01/000002.jpg", cv::IMREAD_GRAYSCALE);
//   assert(!img2.empty());
//
//   Mat flow_mat = flow.get_flow(img2);
//
//   int height = flow_mat.size().height;
//   int width = flow_mat.size().width ;
//   int channels = flow_mat.channels();
//   std::cout << height << ',' << width << ',' << channels << '\n';
//   char *data = (char *)flow_mat.data;
//   std::cout << (data[height*width*channels*sizeof(float)]) << '\n';
//
//   return 0;
// }
