#pragma once
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include <opencv2/core.hpp>


struct ModelOptions {
    size_t num_layers = 4;
    size_t netInputHeight = 128;
    size_t netInputWidth = 128;
    float anchor_offset_x = 0.5;
    float anchor_offset_y = 0.5;
    std::vector<size_t> strides{8, 16, 16, 16};
    float interpolated_scale_aspect_ratio = 1.0;
    float confidence_threshold = 0.5;
    size_t num_of_boxes = 896;
    size_t points = 16;
};

struct Anchor {
    float x_center;
    float y_center;
    Anchor(float x, float y): x_center(x), y_center(y){}
};

struct FaceBox {
    float left;
    float top;
    float right;
    float bottom;
    float confidence;
};

class LiteModel {
public:
    LiteModel(std::unique_ptr<tflite::Interpreter>& interpreter) : interpreter(interpreter) {
        setAnchors();
    };

    std::vector<FaceBox> infer(cv::Mat& input);
private:
    ModelOptions options;
    std::unique_ptr<tflite::Interpreter>& interpreter;
    cv::Size origImageSize;
    std::vector<Anchor> anchors;

    void setAnchors();
    void preprocess(const cv::Mat& input, float* in_tensor);
    std::vector<FaceBox> postprocess(float* prob, float* boxes);
    void decode(float* boxes);
    void sigmoid(float* prob);
    std::vector<FaceBox> getDetections(float* prob, float* boxes);
};
