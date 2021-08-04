# We have proposed Edge YOLO algorithm and compare this algorithm with other object detection network
For this purpose, we provide the Edge YOLO algorithm in the COCO and KITTI data sets related files, including modified network and structure and weight files.
We also provide a Tensorrt version suitable for edge reasoning
## Other object detection network link address

Paper YOLO v4: https://arxiv.org/abs/2004.10934

Paper Scaled YOLO v4: https://arxiv.org/abs/2011.08036  use to reproduce results: [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)

More details in articles on medium:

- [Scaled_YOLOv4](https://alexeyab84.medium.com/scaled-yolo-v4-is-the-best-neural-network-for-object-detection-on-ms-coco-dataset-39dfa22fa982?source=friends_link&sk=c8553bfed861b1a7932f739d26f487c8)
- [YOLOv4](https://medium.com/@alexeyab84/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe?source=friends_link&sk=6039748846bbcf1d960c3061542591d7)

Manual: https://github.com/AlexeyAB/darknet/wiki

Discussion:

- [Reddit](https://www.reddit.com/r/MachineLearning/comments/gydxzd/p_yolov4_the_most_accurate_realtime_neural/)
- [Google-groups](https://groups.google.com/forum/#!forum/darknet)
- [Discord](https://discord.gg/zSq8rtW)

About Darknet framework: http://pjreddie.com/darknet/

[![Darknet Continuous Integration](https://github.com/AlexeyAB/darknet/workflows/Darknet%20Continuous%20Integration/badge.svg)](https://github.com/AlexeyAB/darknet/actions?query=workflow%3A%22Darknet+Continuous+Integration%22)
[![CircleCI](https://circleci.com/gh/AlexeyAB/darknet.svg?style=svg)](https://circleci.com/gh/AlexeyAB/darknet)
[![TravisCI](https://travis-ci.org/AlexeyAB/darknet.svg?branch=master)](https://travis-ci.org/AlexeyAB/darknet)
[![Contributors](https://img.shields.io/github/contributors/AlexeyAB/Darknet.svg)](https://github.com/AlexeyAB/darknet/graphs/contributors)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](https://github.com/AlexeyAB/darknet/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/75388965.svg)](https://zenodo.org/badge/latestdoi/75388965)
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2004.10934-B31B1B.svg)](https://arxiv.org/abs/2004.10934)
[![arxiv.org](http://img.shields.io/badge/cs.CV-arXiv%3A2011.08036-B31B1B.svg)](https://arxiv.org/abs/2011.08036)
[![colab](https://user-images.githubusercontent.com/4096485/86174089-b2709f80-bb29-11ea-9faf-3d8dc668a1a5.png)](https://colab.research.google.com/drive/12QusaaRj_lUwCGDvQNfICpa7kA7_a2dE)
[![colab](https://user-images.githubusercontent.com/4096485/86174097-b56b9000-bb29-11ea-9240-c17f6bacfc34.png)](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg)

- [YOLOv4 model zoo](https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo)
- [Requirements (and how to install dependencies)](#requirements-for-windows-linux-and-macos)
- [Pre-trained models](#pre-trained-models)
- [FAQ - frequently asked questions](https://github.com/AlexeyAB/darknet/wiki/FAQ---frequently-asked-questions)
- [Explanations in issues](https://github.com/AlexeyAB/darknet/issues?q=is%3Aopen+is%3Aissue+label%3AExplanations)
- [Yolo v4 in other frameworks (TensorRT, TensorFlow, PyTorch, OpenVINO, OpenCV-dnn, TVM,...)](#yolo-v4-in-other-frameworks)
- [Datasets](#datasets)

- [Yolo v4, v3 and v2 for Windows and Linux](#yolo-v4-v3-and-v2-for-windows-and-linux)
  - [(neural networks for object detection)](#neural-networks-for-object-detection)
    - [GeForce RTX 2080 Ti](#geforce-rtx-2080-ti)
      - [Youtube video of results](#youtube-video-of-results)
      - [How to evaluate AP of YOLOv4 on the MS COCO evaluation server](#how-to-evaluate-ap-of-yolov4-on-the-ms-coco-evaluation-server)
      - [How to evaluate FPS of YOLOv4 on GPU](#how-to-evaluate-fps-of-yolov4-on-gpu)
      - [Pre-trained models](#pre-trained-models)
    - [Requirements for Windows, Linux and macOS](#requirements-for-windows-linux-and-macos)
    - [Yolo v4 in other frameworks](#yolo-v4-in-other-frameworks)
      - [Datasets](#datasets)
    - [Improvements in this repository](#improvements-in-this-repository)
      - [How to use on the command line](#how-to-use-on-the-command-line)
        - [For using network video-camera mjpeg-stream with any Android smartphone](#for-using-network-video-camera-mjpeg-stream-with-any-android-smartphone)
    - [How to compile on Linux/macOS (using `CMake`)](#how-to-compile-on-linuxmacos-using-cmake)
    - [Using also PowerShell](#using-also-powershell)
    - [How to compile on Linux (using `make`)](#how-to-compile-on-linux-using-make)
    - [How to compile on Windows (using `CMake`)](#how-to-compile-on-windows-using-cmake)
    - [How to compile on Windows (using `vcpkg`)](#how-to-compile-on-windows-using-vcpkg)
  - [How to train with multi-GPU](#how-to-train-with-multi-gpu)
  - [How to train (to detect your custom objects)](#how-to-train-to-detect-your-custom-objects)
    - [How to train tiny-yolo (to detect your custom objects)](#how-to-train-tiny-yolo-to-detect-your-custom-objects)
  - [When should I stop training](#when-should-i-stop-training)
    - [Custom object detection](#custom-object-detection)
  - [How to improve object detection](#how-to-improve-object-detection)
  - [How to mark bounded boxes of objects and create annotation files](#how-to-mark-bounded-boxes-of-objects-and-create-annotation-files)
  - [How to use Yolo as DLL and SO libraries](#how-to-use-yolo-as-dll-and-so-libraries)

![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

![scaled_yolov4](https://user-images.githubusercontent.com/4096485/112776361-281d8380-9048-11eb-8083-8728b12dcd55.png) AP50:95 - FPS (Tesla V100) Paper: https://arxiv.org/abs/2011.08036

----

![modern_gpus](https://user-images.githubusercontent.com/4096485/82835867-f1c62380-9ecd-11ea-9134-1598ed2abc4b.png) AP50:95 / AP50 - FPS (Tesla V100) Paper: https://arxiv.org/abs/2004.10934

tkDNN-TensorRT accelerates YOLOv4 **~2x** times for batch=1 and **3x-4x** times for batch=4.

- tkDNN: https://github.com/ceccocats/tkDNN
- OpenCV: https://gist.github.com/YashasSamaga/48bdb167303e10f4d07b754888ddbdcf

### GeForce RTX 2080 Ti

| Network Size               | Darknet, FPS (avg) | tkDNN TensorRT FP32, FPS | tkDNN TensorRT FP16, FPS | OpenCV FP16, FPS | tkDNN TensorRT FP16 batch=4, FPS | OpenCV FP16 batch=4, FPS | tkDNN Speedup |
|:--------------------------:|:------------------:|-------------------------:|-------------------------:|-----------------:|---------------------------------:|-------------------------:|--------------:|
|320                         | 100                | 116                      | **202**                  | 183              | 423                              | **430**                  | **4.3x**      |
|416                         | 82                 | 103                      | **162**                  | 159              | 284                              | **294**                  | **3.6x**      |
|512                         | 69                 | 91                       | 134                      | **138**          | 206                              | **216**                  | **3.1x**      |
|608                         | 53                 | 62                       | 103                      | **115**          | 150                              | **150**                  | **2.8x**      |
|Tiny 416                    | 443                | 609                      | **790**                  | 773              | **1774**                         | 1353                     | **3.5x**      |
|Tiny 416 CPU Core i7 7700HQ | 3.4                | -                        | -                        | 42               | -                                | 39                       | **12x**       |

- Yolo v4 Full comparison: [map_fps](https://user-images.githubusercontent.com/4096485/80283279-0e303e00-871f-11ea-814c-870967d77fd1.png)
- Yolo v4 tiny comparison: [tiny_fps](https://user-images.githubusercontent.com/4096485/85734112-6e366700-b705-11ea-95d1-fcba0de76d72.png)
- CSPNet: [paper](https://arxiv.org/abs/1911.11929) and [map_fps](https://user-images.githubusercontent.com/4096485/71702416-6645dc00-2de0-11ea-8d65-de7d4b604021.png) comparison: https://github.com/WongKinYiu/CrossStagePartialNetworks
- Yolo v3 on MS COCO: [Speed / Accuracy (mAP@0.5) chart](https://user-images.githubusercontent.com/4096485/52151356-e5d4a380-2683-11e9-9d7d-ac7bc192c477.jpg)
- Yolo v3 on MS COCO (Yolo v3 vs RetinaNet) - Figure 3: https://arxiv.org/pdf/1804.02767v1.pdf
- Yolo v2 on Pascal VOC 2007: https://hsto.org/files/a24/21e/068/a2421e0689fb43f08584de9d44c2215f.jpg
- Yolo v2 on Pascal VOC 2012 (comp4): https://hsto.org/files/3a6/fdf/b53/3a6fdfb533f34cee9b52bdd9bb0b19d9.jpg

#### Youtube video of results

| [![Yolo v4](https://user-images.githubusercontent.com/4096485/101360000-1a33cf00-38ae-11eb-9e5e-b29c5fb0afbe.png)](https://youtu.be/1_SiUOYUoOI "Yolo v4") |  [![Scaled Yolo v4](https://user-images.githubusercontent.com/4096485/101359389-43a02b00-38ad-11eb-866c-f813e96bf61a.png)](https://youtu.be/YDFf-TqJOFE "Scaled Yolo v4") |
|---|---|

Others: https://www.youtube.com/user/pjreddie/videos

#### How to evaluate AP of YOLOv4 on the MS COCO evaluation server

1. Download and unzip test-dev2017 dataset from MS COCO server: http://images.cocodataset.org/zips/test2017.zip
2. Download list of images for Detection tasks and replace the paths with yours: https://raw.githubusercontent.com/AlexeyAB/darknet/master/scripts/testdev2017.txt
3. Download `yolov4.weights` file 245 MB: [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) (Google-drive mirror [yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT) )
4. Content of the file `cfg/coco.data` should be

```ini
classes= 80
train  = <replace with your path>/trainvalno5k.txt
valid = <replace with your path>/testdev2017.txt
names = data/coco.names
backup = backup
eval=coco
```

5. Create `/results/` folder near with `./darknet` executable file
6. Run validation: `./darknet detector valid cfg/coco.data cfg/yolov4.cfg yolov4.weights`
7. Rename the file  `/results/coco_results.json` to `detections_test-dev2017_yolov4_results.json` and compress it to `detections_test-dev2017_yolov4_results.zip`
8. Submit file `detections_test-dev2017_yolov4_results.zip` to the MS COCO evaluation server for the `test-dev2019 (bbox)`

#### How to evaluate FPS of YOLOv4 on GPU

1. Compile Darknet with `GPU=1 CUDNN=1 CUDNN_HALF=1 OPENCV=1` in the `Makefile`
2. Download `yolov4.weights` file 245 MB: [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) (Google-drive mirror [yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT) )
3. Get any .avi/.mp4 video file (preferably not more than 1920x1080 to avoid bottlenecks in CPU performance)
4. Run one of two commands and look at the AVG FPS:

- include video_capturing + NMS + drawing_bboxes:
    `./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights test.mp4 -dont_show -ext_output`
- exclude video_capturing + NMS + drawing_bboxes:
    `./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights test.mp4 -benchmark`
    
1. in C++: https://github.com/AlexeyAB/Yolo_mark
2. in Python: https://github.com/tzutalin/labelImg
3. in Python: https://github.com/Cartucho/OpenLabeling
4. in C++: https://www.ccoderun.ca/darkmark/
5. in JavaScript: https://github.com/opencv/cvat
6. in C++: https://github.com/jveitchmichaelis/deeplabel
7. in C#: https://github.com/BMW-InnovationLab/BMW-Labeltool-Lite
8. DL-Annotator for Windows ($30): [url](https://www.microsoft.com/en-us/p/dlannotator/9nsx79m7t8fn?activetab=pivot:overviewtab)
9. v7labs - the greatest cloud labeling tool ($1.5 per hour): https://www.v7labs.com/

## How to use Yolo as DLL and SO libraries

- on Linux
  - using `build.sh` or
  - build `darknet` using `cmake` or
  - set `LIBSO=1` in the `Makefile` and do `make`
- on Windows
  - using `build.ps1` or
  - build `darknet` using `cmake` or
  - compile `build\darknet\yolo_cpp_dll.sln` solution or `build\darknet\yolo_cpp_dll_no_gpu.sln` solution

There are 2 APIs:

- C API: https://github.com/AlexeyAB/darknet/blob/master/include/darknet.h
  - Python examples using the C API:
    - https://github.com/AlexeyAB/darknet/blob/master/darknet.py
    - https://github.com/AlexeyAB/darknet/blob/master/darknet_video.py

- C++ API: https://github.com/AlexeyAB/darknet/blob/master/include/yolo_v2_class.hpp
  - C++ example that uses C++ API: https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp

----

1. To compile Yolo as C++ DLL-file `yolo_cpp_dll.dll` - open the solution `build\darknet\yolo_cpp_dll.sln`, set **x64** and **Release**, and do the: Build -> Build yolo_cpp_dll
    - You should have installed **CUDA 10.2**
    - To use cuDNN do: (right click on project) -> properties -> C/C++ -> Preprocessor -> Preprocessor Definitions, and add at the beginning of line: `CUDNN;`

2. To use Yolo as DLL-file in your C++ console application - open the solution `build\darknet\yolo_console_dll.sln`, set **x64** and **Release**, and do the: Build -> Build yolo_console_dll

    - you can run your console application from Windows Explorer `build\darknet\x64\yolo_console_dll.exe`
    **use this command**: `yolo_console_dll.exe data/coco.names yolov4.cfg yolov4.weights test.mp4`

    - after launching your console application and entering the image file name - you will see info for each object:
    `<obj_id> <left_x> <top_y> <width> <height> <probability>`
    - to use simple OpenCV-GUI you should uncomment line `//#define OPENCV` in `yolo_console_dll.cpp`-file: [link](https://github.com/AlexeyAB/darknet/blob/a6cbaeecde40f91ddc3ea09aa26a03ab5bbf8ba8/src/yolo_console_dll.cpp#L5)
    - you can see source code of simple example for detection on the video file: [link](https://github.com/AlexeyAB/darknet/blob/ab1c5f9e57b4175f29a6ef39e7e68987d3e98704/src/yolo_console_dll.cpp#L75)

`yolo_cpp_dll.dll`-API: [link](https://github.com/AlexeyAB/darknet/blob/master/src/yolo_v2_class.hpp#L42)

```cpp
struct bbox_t {
    unsigned int x, y, w, h;    // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;        // class of object - from range [0, classes-1]
    unsigned int track_id;        // tracking id for video (0 - untracked, 1 - inf - tracked object)
    unsigned int frames_counter;// counter of frames on which the object was detected
};

class Detector {
public:
        Detector(std::string cfg_filename, std::string weight_filename, int gpu_id = 0);
        ~Detector();

        std::vector<bbox_t> detect(std::string image_filename, float thresh = 0.2, bool use_mean = false);
        std::vector<bbox_t> detect(image_t img, float thresh = 0.2, bool use_mean = false);
        static image_t load_image(std::string image_filename);
        static void free_image(image_t m);

#ifdef OPENCV
        std::vector<bbox_t> detect(cv::Mat mat, float thresh = 0.2, bool use_mean = false);
        std::shared_ptr<image_t> mat_to_image_resize(cv::Mat mat) const;
#endif
};
```
