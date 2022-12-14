#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/trace.hpp>


#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/examples/label_image/log.h"

#include "model.hpp"

const std::string keys =
    "{ help h |                                    | Print this help message }"
    "{ input  |                                    | Path to the input video file }"
    "{ output |                                    | (Optional) path to output video file }"
    "{ num_threads |                               | (Optional) number of threads for tflite }"
    "{ model  |                                    |  }"
    ;


cv::Mat draw_detections(cv::Mat frame, const std::vector<FaceBox>& results) {
    cv::Scalar accepte_color(0, 220, 0);
    cv::Scalar disable_color(0, 0, 255);
    for (auto& result : results) {
        cv::Rect rect(cv::Point2d(result.left, result.top), cv::Point2d(result.right, result.bottom));
        cv::Scalar color = accepte_color;
        cv::rectangle(frame, rect, color, 1);
        auto drawPhotoFrameCorner = [&](cv::Point p, int dx, int dy) {
            cv::line(frame, p, cv::Point(p.x, p.y + dy), color, 2);
            cv::line(frame, p, cv::Point(p.x + dx, p.y), color, 2);
        };

        int dx = static_cast<int>(0.1 * rect.width);
        int dy = static_cast<int>(0.1 * rect.height);

        drawPhotoFrameCorner(rect.tl(), dx, dy);
        drawPhotoFrameCorner(cv::Point(rect.x + rect.width - 1, rect.y), -dx, dy);
        drawPhotoFrameCorner(cv::Point(rect.x, rect.y + rect.height - 1), dx, -dy);
        drawPhotoFrameCorner(cv::Point(rect.x + rect.width - 1, rect.y + rect.height - 1), -dx, -dy);
    }
    return frame;
}


int main(int argc, char** argv) {
    cv::CommandLineParser cmd(argc, argv, keys);

    const std::string model_path = cmd.get<std::string>("model");
    std::cout <<model_path << '\n';
    const auto input_file_name = cmd.get<std::string>("input");
    const std::string output = cmd.get<std::string>("output");
    const int num_threads = std::atoi(cmd.get<std::string>("num_threads").c_str());

    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    // TFLITE_MINIMAL_CHECK(model != nullptr);
    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);

    LOG(INFO) << "tensors size: " << interpreter->tensors_size();
    LOG(INFO) << "nodes size: " << interpreter->nodes_size();
    LOG(INFO) << "inputs: " << interpreter->inputs().size();
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0);
    TfLiteIntArray* dims = interpreter->input_tensor(0)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_channels = dims->data[3];

    LOG(INFO) << "input shape: " << wanted_height << "x" << wanted_width << "x" << wanted_channels;
    std::cout << model->GetModel() << '\n';
    interpreter->SetNumThreads(num_threads > 0 ? num_threads : 1);

    if (input_file_name.empty()) {
        CV_Assert(false && "input_file_name is empty. Please use absolute path for --input=");
    }
    cv::VideoCapture cap(input_file_name);
    cv::Size in_size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Scalar default_color(155, 255, 120);
    cv::Mat background(in_size, CV_8UC3, default_color);

    cv::VideoWriter writer;
    int frames = 0;
    cv::Mat input;

    cv::TickMeter tm;
    tm.start();
    interpreter->AllocateTensors();
    LiteModel inf_model(interpreter);
    while (true) {
        cap.read(input);
        if (input.empty()) {
            break;
        }

        std::vector<FaceBox> results = inf_model.infer(input);
        // std::cout << results[0].left << " " << results[0].top << " " << results[0].right << results[0].bottom << '\n';
        cv::Mat frame = draw_detections(input.clone(), results);

        cv::imshow("Result faces", frame);
        char key = cv::waitKey(1);
        if (key == 27) {
            break;
        }
        ++frames;
    }
    tm.stop();
    std::cout << "Processed " << frames << " frames" << " (" << frames / tm.getTimeSec() << " FPS)" << std::endl;
    return 0;
}
