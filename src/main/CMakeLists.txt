set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(
	SRC_LIST
	main.cpp
	)

include_directories(
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/../lib/nms
	/home/xieqiang/Documents/Code/Detection/py-faster-rcnn-master/caffe-fast-rcnn/include
	${PROJECT_SOURCE_DIR}/lib
	/usr/local/include
	/usr/include
	/usr/include/python2.7
	/opt/intel/mkl/include
	/usr/local/cuda/include
	)

link_directories(
	${PROJECT_SOURCE_DIR}/lib
	/home/xieqiang/Documents/Code/Detection/py-faster-rcnn-master/caffe-fast-rcnn/build/lib
	/home/xieqiang/Documents/Code/Detection/py-faster-rcnn-master/lib/nms
	/usr/local/lib
	/usr/local/lib/python2.7
	/usr/lib
	)

add_executable(
	main
	${SRC_LIST}
	)

target_link_libraries(
	main
	libfaster_rcnn.so
	libcaffe.so
	libgpu_nms.so
	libopencv_highgui.so
	libopencv_core.so
	libopencv_imgproc.so
	libopencv_imgcodecs.so
	libglog.so
	libboost_system.so
	libboost_python.so
	libpython2.7.so
	)
