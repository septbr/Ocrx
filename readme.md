# Ocrx
 Ocr without opencv

## Usage
```cpp
#include "Ocrx/Ocrx.hpp"

int main()
{
    auto ocr = Ocrx::create(
        "PPOCRv3/ch_PP-OCRv3_det.onnx",
        "PPOCRv3/ch_ppocr_mobile_v2.0_cls.onnx",
        "PPOCRv3/ch_PP-OCRv3_rec.onnx",
        "PPOCRv3/ppocr_keys_v1.txt");

    auto input = Ocrx::Image(1200, 900, 3);
    // read input image ...

    const auto results = ocr.recognizes(input, true);
    for (int index = 0; index < results.size(); ++index)
    {
        const auto &result = results[index];
        std::cout << "[ocrx] " << index
                << " score: " << result.score
                << " text: " << result.text << std::endl;
    }
    return 0;
}
```

## Dependencies
- [onnxruntime](https://github.com/microsoft/onnxruntime)

## License
MIT