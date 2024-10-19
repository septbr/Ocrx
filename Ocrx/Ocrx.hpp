#pragma once

#include <vector>
#include <string>

class Ocrx
{
private:
    template <typename T>
    struct _Point
    {
        T x, y;
        _Point(T x = 0, T y = 0) : x(x), y(y) {}
    };

public:
    using Point = _Point<int>;
    using Point2f = _Point<float>;

    struct Image
    {
        int width, height, channel;
        std::vector<unsigned char> data;

        Image(int width, int height, int channel) : width(width), height(height), channel(channel), data(width * height * channel, 0) {}
    };
    struct Result
    {
        Point box[3]; // tl point, horizontal direction, vertical direction
        float score;
        std::string text;
    };

protected:
    const Ocrx *impl;
    explicit Ocrx(Ocrx *impl) : impl(impl) {}
    Ocrx(const Ocrx &) = delete;
    Ocrx &operator=(const Ocrx &) = delete;

public:
    Ocrx(Ocrx &&other) : impl(other.impl) { other.impl = nullptr; }
    Ocrx &operator=(Ocrx &&other)
    {
        delete impl;
        impl = other.impl;
        other.impl = nullptr;
        return *this;
    };
    virtual ~Ocrx() { delete impl; }

    virtual std::vector<Result> recognizes(const Image &rgb_image, bool rotated = false) const { return impl ? impl->recognizes(rgb_image, rotated) : std::vector<Result>{}; }
    virtual Result recognize(const Image &rgb_image) const { return impl ? impl->recognize(rgb_image) : Result{}; }

public:
    static Ocrx create(const char *det_path, const char *cls_path, const char *rec_path, const char *characters_path);
};
