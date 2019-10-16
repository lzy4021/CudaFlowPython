# CudaFlowPython
Compute gpu dense optical flow and then call it in python. On GTX 1080, it takes roughly 6ms per frame.  
It will generate optical flow with size 640x360x2. There are 2 channels because 1 is for horizontal flow, then 1 is for vertical flow.  

Make Sure you installed libboost. And change CmakeLists.txt according to your install path. If you are using libboost1.60+ then you need to modify `cpp_flow.cpp`:
```cpp
#include <boost/numpy.hpp>
namespace np = boost::numpy;
```
to  
```cpp
#include <boost/python/numpy.hpp>
namespace np = boost::python::numpy;
```
