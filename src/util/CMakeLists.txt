set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_C_COMPILER g++)

set(
	LIB_SRC_LIST
	faster_rcnn.cpp
	)

include_directories(
	${PROJECT_SOURCE_DIR}/include
	/home/xieqiang/Documents/Code/Detection/py-faster-rcnn-master/caffe-fast-rcnn/include
	/home/xieqiang/Documents/Code/Detection/py-faster-rcnn-master/caffe-fast-rcnn/include/caffe
	/home/xieqiang/Documents/Code/Detection/py-faster-rcnn-master/lib/nms
	/usr/include
	/usr/local/include
	/usr/include/python2.7
	/usr/local/
	/opt/intel/mkl/include
	/usr/local/cuda/include
	)
add_library(faster_rcnn SHARED ${LIB_SRC_LIST})
