#include "model.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

void LiteModel::setAnchors() {
    assert(options.strides.size() == options.num_layers);
    int layer_id = 0;
    while (layer_id < options.num_layers) {
        int last_same_stride_layer = layer_id;
        int repeats = 0;
        while (last_same_stride_layer < options.num_layers &&
               options.strides[last_same_stride_layer] == options.strides[layer_id]) {
            last_same_stride_layer += 1;
            repeats += 2;
        }
        size_t stride = options.strides[layer_id];
        int feature_map_height = options.netInputHeight / stride;
        int feature_map_width = options.netInputWidth / stride;
        for(int y = 0; y < feature_map_height; ++y) {
            float y_center = (y + options.anchor_offset_y) / feature_map_height;
            for(int x = 0; x < feature_map_width; ++x) {
                float x_center = (x + options.anchor_offset_x) / feature_map_width;
                for(int i = 0; i < repeats; ++i)
                    anchors.emplace_back(x_center, y_center);
            }
        }

        layer_id = last_same_stride_layer;
    }
}


std::vector<FaceBox> LiteModel::infer(cv::Mat& input) {
    float* in_tensor = interpreter->typed_input_tensor<float>(0);

    preprocess(input, in_tensor);
    interpreter->Invoke();
    float* prob = interpreter->typed_output_tensor<float>(1);
    float* res_output = interpreter->typed_output_tensor<float>(0);
    return postprocess(prob, res_output);
}

void LiteModel::preprocess(const cv::Mat& input, float* in_tensor) {
    origImageSize = input.size();
    double scale = std::min(static_cast<double>(options.netInputWidth) / input.cols,
                            static_cast<double>(options.netInputHeight) / input.rows);
    cv::Mat resizedImage;
    cv::resize(input, resizedImage, cv::Size(options.netInputWidth, options.netInputHeight));
    // cv::resize(input, resizedImage, cv::Size(0, 0), scale, scale);
    // int dx = (options.netInputWidth - resizedImage.cols) / 2;
    // int dy = (options.netInputHeight - resizedImage.rows) / 2;
    // cv::Mat dst;
    // cv::copyMakeBorder(resizedImage, dst, dy, options.netInputHeight - resizedImage.rows - dy,
    //     dx, options.netInputWidth - resizedImage.cols - dx, cv::BORDER_CONSTANT, cv::Scalar(0.));
    cv::Mat cvt;
    resizedImage.convertTo(cvt, CV_32F);
    cv::cvtColor(cvt, cvt, cv::COLOR_BGR2RGB);
    cvt -= cv::Scalar(127.5, 127.5, 127.5);
    cvt /= cv::Scalar(127.5, 127.5, 127.5);
    float* cvt_ptr = cvt.ptr<float>();

    for (int32_t i = 0; i < options.netInputHeight * options.netInputWidth; i++) {
        for (int32_t c = 0; c < 3; c++) {
            in_tensor[i * 3 + c] = cvt_ptr[i * 3 + c];
        }
    }
}

std::vector<int> nms(const std::vector<FaceBox>& boxes, const float thresh=0.6, bool includeBoundaries=false) {
    std::vector<float> scores(boxes.size());
    for(int i = 0; i < boxes.size(); ++i) {
        scores[i] = boxes[i].confidence;
    }
    std::vector<float> areas(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        areas[i] = (boxes[i].right - boxes[i].left + includeBoundaries) * (boxes[i].bottom - boxes[i].top + includeBoundaries);
    }
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&scores](int o1, int o2) { return scores[o1] > scores[o2]; });

    size_t ordersNum = 0;
    for (; ordersNum < order.size() && scores[order[ordersNum]] >= 0; ordersNum++);

    std::vector<int> keep;
    bool shouldContinue = true;
    for (size_t i = 0; shouldContinue && i < ordersNum; ++i) {
        auto idx1 = order[i];
        if (idx1 >= 0) {
            keep.push_back(idx1);
            shouldContinue = false;
            for (size_t j = i + 1; j < ordersNum; ++j) {
                auto idx2 = order[j];
                if (idx2 >= 0) {
                    shouldContinue = true;
                    auto overlappingWidth = std::fminf(boxes[idx1].right, boxes[idx2].right) - std::fmaxf(boxes[idx1].left, boxes[idx2].left);
                    auto overlappingHeight = std::fminf(boxes[idx1].bottom, boxes[idx2].bottom) - std::fmaxf(boxes[idx1].top, boxes[idx2].top);
                    auto intersection = overlappingWidth > 0 && overlappingHeight > 0 ? overlappingWidth * overlappingHeight : 0;
                    auto overlap = intersection / (areas[idx1] + areas[idx2] - intersection);

                    if (overlap >= thresh) {
                        order[j] = -1;
                    }
                }
            }
        }
    }
    return keep;
}

std::vector<FaceBox> LiteModel::postprocess(float* prob, float* boxes) {
    sigmoid(prob);
    decode(boxes);
    std::vector<FaceBox> detections = getDetections(prob, boxes);
    std::vector<int> keep = nms(detections);
    std::vector<FaceBox> results;
    for(auto& index : keep) {
        results.push_back(detections[index]);
    }
    return results;
}

std::vector<FaceBox> LiteModel::getDetections(float* prob, float* boxes){
    std::vector<FaceBox> detections;
    for(int box_index = 0; box_index < options.num_of_boxes; ++box_index) {
        float score = prob[box_index];

        if (score < options.confidence_threshold)
            continue;

        FaceBox detected_object;
        detected_object.confidence = score;

        const int start_pos = box_index * options.points;
        const float x0 = std::min(std::max(0.0f, boxes[start_pos]), 1.0f) * origImageSize.width;
        const float y0 = std::min(std::max(0.0f, boxes[start_pos + 1]), 1.0f) * origImageSize.height;
        const float x1 = std::min(std::max(0.0f, boxes[start_pos + 2]), 1.0f) * origImageSize.width;
        const float y1 = std::min(std::max(0.0f, boxes[start_pos + 3]), 1.0f) * origImageSize.height;
        detected_object.left = static_cast<int>(round(static_cast<double>(x0)));
        detected_object.top  = static_cast<int>(round(static_cast<double>(y0)));
        detected_object.right = static_cast<int>(round(static_cast<double>(x1)));
        detected_object.bottom = static_cast<int>(round(static_cast<double>(y1)));
        detections.emplace_back(detected_object);
    }

    return detections;
}

void LiteModel::sigmoid(float* prob) {
    for(int i = 0; i < options.num_of_boxes; ++i) {
       prob[i] = 1.f / (1.f + exp(-prob[i]));
    }
}

void LiteModel::decode(float* boxes) {
    for(int i = 0; i < options.num_of_boxes; ++i) {
        size_t scale = options.netInputHeight;
        size_t num_points = options.points / 2;
        const int start_pos = i * options.points;
        for(int j = 0; j < num_points; ++j) {
            boxes[start_pos + 2*j]  = boxes[start_pos + 2*j]  / scale;
            boxes[start_pos + 2*j + 1]  = boxes[start_pos + 2*j + 1] / scale;
            if (j != 1) {
                boxes[start_pos + 2*j] += anchors[i].x_center;
                boxes[start_pos + 2*j + 1] += anchors[i].y_center;
            }
        }

        // convert x_center, y_center, w, h to xmin, ymin, xmax, ymax

        float half_width = boxes[start_pos + 2] / 2;
        float half_height = boxes[start_pos + 3] / 2;
        float center_x = boxes[start_pos];
        float center_y = boxes[start_pos + 1];

        boxes[start_pos] -= half_width;
        boxes[start_pos + 1] -= half_height;

        boxes[start_pos + 2] = center_x + half_width;
        boxes[start_pos + 3] = center_y + half_height;
    }
}
