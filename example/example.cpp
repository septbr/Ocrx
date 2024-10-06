
#ifdef _WIN32
#include <windows.h>
#endif

#include "Ocrx.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    // auto path = "E:/ocrx/example/images/ocr_test2.png";
    auto path = "E:/ocrx/example/images/ocr_test3.png";

    {
        auto ocr = Ocrx::create(
            "E:/ocrx/models/PPOCRv3/ch_PP-OCRv3_det.onnx",
            "E:/ocrx/models/PPOCRv3/ch_ppocr_mobile_v2.0_cls.onnx",
            "E:/ocrx/models/PPOCRv3/ch_PP-OCRv3_rec.onnx",
            "E:/ocrx/models/PPOCRv3/ppocr_keys_v1.txt");

        auto rgb_image = cv::imread(path);
        cv::cvtColor(rgb_image, rgb_image, cv::COLOR_BGR2RGB);
        auto input = Ocrx::Image(rgb_image.cols, rgb_image.rows, 3);
        for (auto index = 0; index < input.data.size(); ++index)
            input.data[index] = rgb_image.data[index];
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        const auto results = ocr.recognizes(input, true);
        std::cout << "ocr v3 recognizes time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "ms" << std::endl;
        for (int index = 0; index < results.size(); ++index)
        {
            const auto &result = results[index];
            std::cout << "[ocrx] " << index
                      //   << " rect: " << result.box[0] << " " << result.box[1] << " " << result.box[2] << " " << result.box[3]
                      << " score: " << result.score
                      << " text: " << result.text << std::endl;

            const auto &box = result.box;
            cv::line(rgb_image, {box[0].x, box[0].y}, {box[0].x + box[1].x, box[0].y + box[1].y}, cv::Scalar(0, 0, 255));
            cv::line(rgb_image, {box[0].x, box[0].y}, {box[0].x + box[2].x, box[0].y + box[2].y}, cv::Scalar(0, 255, 0));
        }
        cv::imshow("rgb_image v3", rgb_image);
    }
    {
        auto ocr = Ocrx::create(
            "E:/ocrx/models/PPOCRv4/ch_PP-OCRv4_det.onnx",
            "E:/ocrx/models/PPOCRv4/ch_ppocr_mobile_v2.0_cls.onnx",
            "E:/ocrx/models/PPOCRv4/ch_PP-OCRv4_rec.onnx",
            "E:/ocrx/models/PPOCRv4/ppocr_keys_v1.txt");

        auto rgb_image = cv::imread(path);
        cv::cvtColor(rgb_image, rgb_image, cv::COLOR_BGR2RGB);
        auto input = Ocrx::Image(rgb_image.cols, rgb_image.rows, 3);
        for (auto index = 0; index < input.data.size(); ++index)
            input.data[index] = rgb_image.data[index];
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        const auto results = ocr.recognizes(input, true);
        std::cout << "ocr v4 recognize time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() << "ms" << std::endl;
        for (int index = 0; index < results.size(); ++index)
        {
            const auto &result = results[index];
            std::cout << "[ocrx] " << index
                      //   << " rect: " << result.box[0] << " " << result.box[1] << " " << result.box[2] << " " << result.box[3]
                      << " score: " << result.score
                      << " text: " << result.text << std::endl;

            const auto &box = result.box;
            cv::line(rgb_image, {box[0].x, box[0].y}, {box[0].x + box[1].x, box[0].y + box[1].y}, cv::Scalar(0, 0, 255));
            cv::line(rgb_image, {box[0].x, box[0].y}, {box[0].x + box[2].x, box[0].y + box[2].y}, cv::Scalar(0, 255, 0));
        }
        cv::imshow("rgb_image v4", rgb_image);
    }

    cv::waitKey(0);
    return 0;
}
