This project is Faster-rcnn detector C++ version demo, if you want to learn more about Faster-rcnn, please click [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn "py-faster-rcnn").


Before using this demo, you need to install the py-faster-rcnn code and compile lib and caffe.Now let's assume that you have installed and compiled caffe and lib in py-faster-rcnn, and the main folder for py-faster-rcnn is assumed to be $ FASTER-RCNN.

**1 Clone the project repository**


    cd $FASTER_RCNN
    git clone https://github.com/QiangXie/libfaster_rcnn_cpp

**2 Rename gpu_nms.so**

Without this step, the compiler will report that the gpu_nms.so file could not be found.

    cd $FASTER-RCNN/libfaster_rcnn_cpp
    mkdir lib
    cp $FASTER-RCNN/lib/nms/gpu_nms.so  $FASTER-RCNN/libfaster_rcnn_cpp/lib/libgpu_nms.so


**3 Build this project**


You must make sure that all dependent libraries have the correct path. You can check `src/main/CmakeLists.txt` and `src/util/CmakeLists.txt` to confirm that.Then

    mkdir build
    cd build
    cmake ..
    make

 
**4 Set the python module path**

The detector uses caffe's Python layer and some related Python programs, so you must ensure the following two paths have been written to your .zshrc or .bashrc:


    export PYTHONPATH=$PYTHONPATH:/home/xieqiang/Documents/Code/Detection/py-faster-rcnn-master/lib
    export PYTHONPATH=$PYTHONPATH:/home/xieqiang/Documents/Code/Detection/py-faster-rcnn-master/caffe-fast-rcnn/python
	


**5 Run the program**

    cd bin
    ./main

This program will detect test1.jpg in bin folder, and print the detected vehicle bounding box, then rectangle bounding box and saved as test.jpg. If you need modify this project to do more, see main.cpp.

That's all!