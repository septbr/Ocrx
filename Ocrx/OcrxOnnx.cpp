#include "Ocrx.hpp"

#include <fstream>
#include <onnxruntime_cxx_api.h>

namespace Utils
{
    using Point = Ocrx::Point;
    using Point2f = Ocrx::Point2f;
    using Image = Ocrx::Image;

    // anticlockwise
    void minAreaRect(const std::vector<Point> &convex_hull, Point2f rect[4])
    {
        rect[0].x = rect[3].x = convex_hull[0].x;
        rect[0].y = rect[3].y = convex_hull[0].y;
        rect[1].x = rect[2].x = convex_hull[1].x;
        rect[1].y = rect[2].y = convex_hull[1].y;
        float min_area = -1;
        float min_wlx = rect[0].x, min_wly = rect[0].y, min_wrx = rect[1].x, min_wry = rect[1].y, min_hx = 0, min_hy = 0;
        for (std::vector<Point>::size_type size = convex_hull.size(), index = 1; index < size; ++index)
        {
            /*
                ab ap aq
                ab * ap = |ab| * |ap| * cos(alpha)
                |aq| = |ap| * cos(alpha)
                |aq| / |ab| = |ap| * cos(alpha) / |ab| = ab * ap / (|ab| * |ab|)
                S = |ab| * |qp|
             */
            const auto &a = convex_hull[index - 1], &b = convex_hull[index];
            float ab2 = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);
            float abx = b.x - a.x, aby = b.y - a.y;
            float wlx = a.x, wly = a.y, wrx = b.x, wry = b.y, hx = 0, hy = 0;
            float dl = 0, dr = 1;
            for (auto offset = 1; offset < size - 1; ++offset)
            {
                const auto &p = convex_hull[(index + offset) % size];
                auto d = (abx * (p.x - a.x) + aby * (p.y - a.y)) / ab2;
                auto qx = a.x + d * abx, qy = a.y + d * aby;
                auto qpx = p.x - qx, qpy = p.y - qy;

                if (qpx * qpx + qpy * qpy > hx * hx + hy * hy)
                    hx = qpx, hy = qpy;

                if (d < dl)
                    dl = d, wlx = qx, wly = qy;
                else if (d > dr)
                    dr = d, wrx = qx, wry = qy;
            }
            auto area = std::sqrt((wrx - wlx) * (wrx - wlx) + (wry - wly) * (wry - wly)) * std::sqrt(hx * hx + hy * hy);
            if (min_area < 0 || area < min_area)
            {
                min_area = area;
                min_wlx = wlx, min_wly = wly, min_wrx = wrx, min_wry = wry, min_hx = hx, min_hy = hy;
            }
        }
        rect[0].x = min_wlx, rect[0].y = min_wly;
        rect[1].x = min_wrx, rect[1].y = min_wry;
        rect[2].x = min_wrx + min_hx, rect[2].y = min_wry + min_hy;
        rect[3].x = min_wlx + min_hx, rect[3].y = min_wly + min_hy;
    }
    enum class ContourType
    {
        Hull,
        Convex,
        Rect,
    };
    // anticlockwise
    std::vector<std::vector<Point>> contours(Image &binary_image, ContourType contour_type = ContourType::Hull)
    {
        std::vector<std::vector<Point>> contours;

        auto width = binary_image.width, height = binary_image.height;
        auto &data = binary_image.data;
        /*
            0 7 6
            1 x 5
            2 3 4
        */
        int direction[] = {-1, -1, -1, 0, -1, 1, 0, 1, 1, 1, 1, 0, 1, -1, 0, -1};
        for (auto y = 0; y < height; ++y)
        {
            for (auto x = 0; x < width; ++x)
            {
                auto index = static_cast<std::size_t>(y) * width + x;
                if (data[index] == 1 && (x == 0 || data[index - 1] == 0))
                {
                    std::vector<Point> contour;
                    for (auto cur_x = x - 1, cur_y = y - 1, cur_dir = 4, step = 0; step < 8; ++step)
                    {
                        auto dir = direction + (cur_dir + step) * 2 % 16;
                        auto next_x = cur_x + *dir, next_y = cur_y + *(dir + 1);
                        if ((next_x >= 0 && next_x < width && next_y >= 0 && next_y < height))
                        {
                            auto &value = data[next_y * width + next_x];
                            if (value != 0)
                            {
                                auto contour_size = contour.size();
                                if (next_x == x && next_y == y && contour_size > 0)
                                    break;
                                if (value == 1)
                                {
                                    value = 2;
                                    Point point(next_x, next_y);
                                    switch (contour_type)
                                    {
                                    case ContourType::Rect:
                                        if (contour_size == 0)
                                        {
                                            contour.push_back(point);
                                            contour.push_back(point);
                                            contour.push_back(point);
                                            contour.push_back(point);
                                        }
                                        else
                                        {
                                            if (point.x > contour[2].x)
                                                contour[2].x = contour[3].x = point.x;
                                            else if (point.x < contour[0].x)
                                                contour[0].x = contour[1].x = point.x;
                                            if (point.y > contour[2].y)
                                                contour[1].y = contour[2].y = point.y;
                                            else if (point.y < contour[0].y)
                                                contour[0].y = contour[3].y = point.y;
                                        }
                                        break;
                                    case ContourType::Convex:
                                        for (decltype(contour_size) index = 0; index < contour_size; ++index)
                                        {
                                            auto peek = index + 1;
                                            while (peek < contour_size)
                                            {
                                                const auto &l = contour[index], &m = contour[peek], &r = point;
                                                if ((l.x - m.x) * (r.y - m.y) - (l.y - m.y) * (r.x - m.x) > 0)
                                                    break;
                                                ++peek;
                                            }
                                            if (peek == contour_size)
                                            {
                                                contour.resize(index + 1);
                                                break;
                                            }
                                        }
                                    default:
                                        contour.push_back(point);
                                        break;
                                    }
                                }
                                cur_x = next_x, cur_y = next_y, cur_dir = (cur_dir + step + 5) % 8, step = 0;
                            }
                        }
                    }
                    contours.push_back(std::move(contour));
                }
            }
        }
        return contours;
    }
    void pixel(const Image &input, float x, float y, unsigned char color[])
    {
        if (x < 0 || y < 0 || x > input.width - 1 || y > input.height - 1)
        {
            std::memset(color, 0, input.channel);
            return;
        }
        auto xl = static_cast<int>(x), yl = static_cast<int>(y);
        auto xr = x > xl ? xl + 1 : xl, yr = y > yl ? yl + 1 : yl;
        auto scale_x = xr - x, scale_y = yr - y;
        for (auto width = input.width, channel = input.channel, c = 0; c < channel; ++c)
        {
            auto v = input.data[(yl * width + xl) * channel + c] * scale_x * scale_y;
            v += input.data[(yl * width + xr) * channel + c] * (1 - scale_x) * scale_y;
            v += input.data[(yr * width + xl) * channel + c] * scale_x * (1 - scale_y);
            v += input.data[(yr * width + xr) * channel + c] * (1 - scale_x) * (1 - scale_y);
            color[c] = static_cast<unsigned char>(v);
        }
    }
    Image resize(const Image &input, int width, int height)
    {
        Image output(width, height, input.channel);
        auto scale_x = static_cast<float>(input.width) / width, scale_y = static_cast<float>(input.height) / height;
        for (auto channel = input.channel, y = 0; y < height; ++y)
        {
            for (auto x = 0; x < width; ++x)
                pixel(input, x * scale_x, y * scale_y, output.data.data() + (y * width + x) * channel);
        }
        return output;
    }
    void rotate180(Image &input)
    {
        for (auto width = input.width, height = input.height, channel = input.channel, end_y = height / 2 + height % 2, y = 0; y < end_y; ++y)
        {
            for (auto end_x = y < height / 2 ? width : width / 2, x = 0; x < end_x; ++x)
            {
                for (auto c = 0; c < channel; ++c)
                    std::swap(input.data[(y * width + x) * channel + c], input.data[((height - 1 - y) * width + (width - 1 - x)) * channel + c]);
            }
        }
    }

    Image limit(const Image &input, int width, int height, bool width_limit = false, bool height_limit = false)
    {
        auto width_to = input.width, height_to = input.height;
        auto width_ratio = 1.0, height_ratio = 1.0;
        if (width > 0 && (!width_limit || width_to > width))
            width_ratio = static_cast<double>(width) / width_to;
        if (height > 0 && (!height_limit || height_to > height))
            height_ratio = static_cast<double>(height) / height_to;
        if (width_limit && height_limit)
            width_ratio = height_ratio = std::min(width_ratio, height_ratio);
        else if (width_limit != height_limit)
        {
            auto ratio = width_limit ? height_ratio : width_ratio;
            if (width_limit)
            {
                width_ratio = width > 0 && static_cast<double>(width_to * ratio) > width ? static_cast<double>(width) / width_to : ratio;
                height_ratio = ratio;
            }
            else
            {
                width_ratio = ratio;
                height_ratio = height > 0 && static_cast<double>(height_to * ratio) > height ? static_cast<double>(height) / height_to : ratio;
            }
        }
        width_to = static_cast<int>(width_to * width_ratio);
        height_to = static_cast<int>(height_to * height_ratio);

        auto output = resize(input, width_to, height_to);
        return output;
    }
    void input_data(const Image &rgb_input, int input_width, int input_height, const float mean[3], const float norm[3], float *data)
    {
        auto input_size = static_cast<std::size_t>(input_width) * input_height * 3;
        for (auto width = rgb_input.width, height = rgb_input.height, channel = rgb_input.channel, y = 0; y < height; ++y)
        {
            for (auto x = 0; x < width; ++x)
            {
                for (int ch = 0; ch < 3; ++ch)
                {
                    auto index = input_width * input_height * ch + y * input_width + x;
                    if (index < input_size)
                    {
                        auto value = (rgb_input.data[(y * width + x) * channel + ch] / 255.0 - mean[ch]) / norm[ch];
                        data[index] = static_cast<float>(value);
                    }
                }
            }
        }
    }
    Image scoop(const Image &input, int origin_x, int origin_y, int width, int height)
    {
        auto output = Image(width, height, input.channel);
        for (auto channel = output.channel, y = 0; y < output.height; ++y)
        {
            for (auto x = 0; x < output.width; ++x)
            {
                for (auto i = (origin_x + x + (origin_y + y) * input.width) * channel, o = (x + y * output.width) * channel, c = 0; c < channel; ++c)
                    output.data[o + c] = i >= 0 && i + c < input.data.size() ? input.data[i + c] : 0;
            }
        }
        return output;
    }
    Image scoop(const Image &input, int origin_x, int origin_y, int width, int height, const Point2f &horizontal, const Point2f &vertical)
    {
        auto output = Image(width, height, input.channel);
        for (auto channel = output.channel, y = 0; y < output.height; ++y)
        {
            for (auto x = 0; x < output.width; ++x)
                pixel(input, origin_x + horizontal.x * x + vertical.x * y, origin_y + horizontal.y * x + vertical.y * y, output.data.data() + (x + y * output.width) * channel);
        }
        return output;
    }
    void split_with_rotate(const Image &input, Image &binary, int side_limit, float up_threshold, float box_threshold, float unclip_ratio, float width_ratio, float height_ratio, std::vector<Point> &boxes, std::vector<Image> &images)
    {
        Point2f rect[4];
        float offset, width, height, origin_x, origin_y;
        Point2f horizontal, vertical;
        for (const auto &contour : Utils::contours(binary, Utils::ContourType::Convex))
        {
            if (contour.size() < 3)
                continue;
            Utils::minAreaRect(contour, rect); // anticlockwise

            auto ul_index = 0;
            for (auto index = 0; index < 4; ++index)
            {
                if (rect[index].y < rect[ul_index].y - up_threshold || (rect[index].y < rect[ul_index].y + up_threshold && rect[index].x < rect[ul_index].x))
                    ul_index = index;
            }
            horizontal.x = rect[(ul_index + 3) % 4].x - rect[ul_index].x, horizontal.y = rect[(ul_index + 3) % 4].y - rect[ul_index].y;
            vertical.x = rect[(ul_index + 1) % 4].x - rect[ul_index].x, vertical.y = rect[(ul_index + 1) % 4].y - rect[ul_index].y;
            width = std::sqrt(horizontal.x * horizontal.x + horizontal.y * horizontal.y), height = std::sqrt(vertical.x * vertical.x + vertical.y * vertical.y);
            if (width < side_limit || height < side_limit)
                continue;

            horizontal.x /= width, horizontal.y /= width;
            vertical.x /= height, vertical.y /= height;
            if (width * box_threshold < height)
            {
                ul_index = (ul_index + 1) % 4;
                std::swap(width, height);
                std::swap(horizontal.x, vertical.x);
                std::swap(horizontal.y, vertical.y);
                horizontal.x *= -1, horizontal.y *= -1;
            }
            offset = width * height / (width + height) * unclip_ratio;
            origin_x = (rect[ul_index].x - (horizontal.x + vertical.x) * offset / 2) * width_ratio, origin_y = (rect[ul_index].y - (horizontal.y + vertical.y) * offset / 2) * height_ratio;
            width = (width + offset) * width_ratio, height = (height + offset) * height_ratio;

            // clockwise
            boxes.push_back({static_cast<int>(origin_x), static_cast<int>(origin_y)});
            boxes.push_back({static_cast<int>(std::ceil(horizontal.x * width)), static_cast<int>(std::ceil(horizontal.y * width))});
            boxes.push_back({static_cast<int>(std::ceil(vertical.x * height)), static_cast<int>(std::ceil(vertical.y * height))});
            images.push_back(scoop(input, static_cast<int>(origin_x), static_cast<int>(origin_y), static_cast<int>(std::ceil(width)), static_cast<int>(std::ceil(height)), horizontal, vertical));
        }
    }
    void split(const Image &input, Image &binary, int side_limit, float up_threshold, float box_threshold, float unclip_ratio, float width_ratio, float height_ratio, std::vector<Point> &boxes, std::vector<Image> &images)
    {
        float offset, width, height, origin_x, origin_y;
        for (const auto &rect : Utils::contours(binary, Utils::ContourType::Rect))
        {
            int width = rect[2].x - rect[0].x, height = rect[2].y - rect[0].y;
            if (width < side_limit || height < side_limit)
                continue;

            offset = width * height / (width + height) * unclip_ratio;
            origin_x = (rect[0].x - offset / 2) * width_ratio, origin_y = (rect[0].y - offset / 2) * height_ratio;
            width = (width + offset) * width_ratio, height = (height + offset) * height_ratio;

            // clockwise
            boxes.push_back({static_cast<int>(origin_x), static_cast<int>(origin_y)});
            boxes.push_back({static_cast<int>(std::ceil(width)), 0});
            boxes.push_back({0, static_cast<int>(std::ceil(height))});
            images.push_back(scoop(input, static_cast<int>(origin_x), static_cast<int>(origin_y), static_cast<int>(std::ceil(width)), static_cast<int>(std::ceil(height))));
        }
    }
}

class OcrxOnnx : public Ocrx
{
    class Segment
    {
    protected:
        Ort::Session session;
        std::vector<Ort::AllocatedStringPtr> input_names_ptr;
        std::vector<Ort::AllocatedStringPtr> output_names_ptr;
        std::vector<const char *> input_names;
        std::vector<const char *> output_names;

    public:
        Segment(const Ort::Env &env, const Ort::SessionOptions &options, const ORTCHAR_T *path) : session(env, path, options)
        {
            Ort::AllocatorWithDefaultOptions allocator;
            for (std::size_t count = session.GetInputCount(), index = 0; index < count; ++index)
            {
                input_names_ptr.push_back(std::move(session.GetInputNameAllocated(index, allocator)));
                input_names.push_back(input_names_ptr[index].get());
            }
            for (std::size_t count = session.GetOutputCount(), index = 0; index < count; index++)
            {
                output_names_ptr.push_back(std::move(session.GetOutputNameAllocated(index, allocator)));
                output_names.push_back(output_names_ptr[index].get());
            }
        }
    };
    class Detector : public Segment
    {
    private:
        const float mean[3]{0.485f, 0.456f, 0.406f}, norm[3]{0.229f, 0.224f, 0.225f};

    public:
        const int input_side_limit = 960, box_side_limit = 3;
        float binary_threshold = 0.3f, up_threshold = 5.0f, box_threshold = 1.5f, unclip_ratio = 2.0f;

        using Segment::Segment;
        void process(const Image &rgb_input, std::vector<Point> &boxes, std::vector<Image> &images, bool rotated = false)
        {
            const auto input_image = Utils::limit(rgb_input, input_side_limit, input_side_limit, true, true);
            const auto input_width = std::max(static_cast<int>(std::round(input_image.width / 32.0) * 32), 32);
            const auto input_height = std::max(static_cast<int>(std::round(input_image.height / 32.0) * 32), 32);
            auto input_data = std::vector<float>(static_cast<std::vector<float>::size_type>(input_width) * input_height * 3, 0);
            Utils::input_data(input_image, input_width, input_height, mean, norm, input_data.data());

            int64_t input_shape[4] = {1, 3, input_height, input_width};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape, sizeof(input_shape) / sizeof(input_shape[0]));
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            const auto output_data = output_tensors[0].GetTensorMutableData<float>();

            const auto width_ratio = static_cast<float>(rgb_input.width) / input_image.width;
            const auto height_ratio = static_cast<float>(rgb_input.height) / input_image.height;
            Image output_image(output_shape[3], output_shape[2], 1);
            for (std::vector<unsigned char>::size_type size = output_image.data.size(), index = 0; index < size; ++index)
            {
                unsigned char data = output_data[index] > binary_threshold ? 1 : 0;
                output_image.data[index] = data;
            }
            auto split = rotated ? Utils::split_with_rotate : Utils::split;
            split(rgb_input, output_image, box_side_limit, up_threshold, box_threshold, unclip_ratio, width_ratio, height_ratio, boxes, images);
        }
    };
    class Classifier : public Segment
    {
    private:
        const float mean[3]{0.5f, 0.5f, 0.5f}, norm[3]{0.5f, 0.5f, 0.5f};
        const int batch = 1, input_width = 192, input_height = 48;

    public:
        float threshold = 0.8f;

        using Segment::Segment;
        void process(std::vector<Image> &rgb_inputs)
        {
            auto input_data = std::vector<float>(input_width * input_height * 3 * rgb_inputs.size(), 0);
            for (std::vector<Image>::size_type size = rgb_inputs.size(), index = 0; index < size; index += batch)
            {
                auto pages = std::min<decltype(size)>(batch, size - index);
                auto input_data_ptr = input_data.data() + input_width * input_height * 3 * index;
                for (auto page = 0; page < pages; ++page)
                {
                    const auto input_image = Utils::limit(rgb_inputs[index + page], input_width, input_height, true);
                    Utils::input_data(input_image, input_width, input_height, mean, norm, input_data_ptr + input_width * input_height * 3 * page);
                }

                int64_t input_shape[4]{static_cast<int64_t>(pages), 3, input_height, input_width};
                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
                auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data_ptr, input_width * input_height * 3 * pages, input_shape, sizeof(input_shape) / sizeof(input_shape[0]));
                auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());
                auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
                auto output_data = output_tensors[0].GetTensorMutableData<float>();

                for (decltype(size) pages = output_shape[0], length = output_shape[1], page = 0; page < pages; ++page)
                {
                    int label = 0;
                    float score = 0.0f;
                    for (decltype(size) index = 0; index < length; ++index)
                    {
                        auto value = output_data[length * page + index];
                        if (value > score)
                        {
                            score = value;
                            label = index;
                        }
                    }
                    if (label == 1 && score > threshold)
                        Utils::rotate180(rgb_inputs[index + page]);
                }
            }
        }
    };
    class Recognizer : public Segment
    {
    private:
        const float mean[3]{0.5f, 0.5f, 0.5f}, norm[3]{0.5f, 0.5f, 0.5f};
        const int batch = 1, input_width = 460, input_height = 48;
        std::vector<std::string> characters;

        void output_characters(const float *input_data, int64_t map, int64_t length, float &score, std::string &text) const
        {
            auto count = 0;
            for (decltype(length) last_label = 0, len = 0; len < length; ++len)
            {
                decltype(map) label = 0;
                auto value = -1.f;
                auto map_data = input_data + map * len;
                for (decltype(map) index = 0; index < map; ++index)
                {
                    auto data = map_data[index];
                    if (data > value)
                    {
                        value = data;
                        label = index;
                    }
                }
                if (label > 0 && label != last_label)
                {
                    text += characters[label - 1];
                    score += value;
                    count += 1;
                }
                last_label = label;
            }
            score = count > 0 ? score / count : -1;
        }

    public:
        Recognizer(const Ort::Env &env, const Ort::SessionOptions &options, const ORTCHAR_T *path, const ORTCHAR_T *characters_path) : Segment(env, options, path)
        {
            std::ifstream file(characters_path);
            std::string line;
            while (std::getline(file, line))
                characters.push_back(line);
            characters.push_back(" ");
        }
        void process(const Image &rgb_input, float &score, std::string &text)
        {
            const auto input_image = Utils::limit(rgb_input, input_width, input_height, true);
            auto input_data = std::vector<float>(static_cast<std::vector<float>::size_type>(input_width) * input_height * 3, 0);
            Utils::input_data(input_image, input_width, input_height, mean, norm, input_data.data());

            int64_t input_shape[4]{1, 3, input_height, input_width};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape, sizeof(input_shape) / sizeof(input_shape[0]));
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            auto output_data = output_tensors[0].GetTensorMutableData<float>();

            output_characters(output_data, output_shape[2], output_shape[1], score, text);
        }
        void process(const std::vector<Image> &rgb_inputs, std::vector<float> &scores, std::vector<std::string> &texts)
        {
            auto input_data = std::vector<float>(input_width * input_height * 3 * rgb_inputs.size(), 0);
            for (std::vector<Image>::size_type size = rgb_inputs.size(), index = 0; index < size; index += batch)
            {
                auto pages = std::min<decltype(size)>(batch, size - index);
                auto input_data_ptr = input_data.data() + input_width * input_height * 3 * index;
                for (auto page = 0; page < pages; ++page)
                {
                    const auto input_image = Utils::limit(rgb_inputs[index + page], input_width, input_height, true);
                    Utils::input_data(input_image, input_width, input_height, mean, norm, input_data_ptr + input_width * input_height * 3 * page);
                }

                int64_t input_shape[4]{static_cast<int64_t>(pages), 3, input_height, input_width};
                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
                auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data_ptr, input_width * input_height * 3 * pages, input_shape, sizeof(input_shape) / sizeof(input_shape[0]));
                auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, input_names.size(), output_names.data(), output_names.size());
                auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
                auto output_data = output_tensors[0].GetTensorMutableData<float>();

                for (decltype(size) pages = output_shape[0], length = output_shape[1], map = output_shape[2], page = 0; page < pages; ++page)
                {
                    auto score = 0.0f;
                    std::string text;
                    output_characters(output_data + map * length * page, map, length, score, text);
                    scores.push_back(score);
                    texts.push_back(std::move(text));
                }
            }
        }
    };

private:
    Ort::Env env;
    Ort::SessionOptions options;
    Detector *detector;
    Classifier *classifier;
    Recognizer *recognizer;

public:
    OcrxOnnx(const ORTCHAR_T *det_path, const ORTCHAR_T *cls_path, const ORTCHAR_T *rec_path, const ORTCHAR_T *characters_path)
        : Ocrx(nullptr),
          env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Ocrx")
    {
        // try
        // {
        //     OrtCUDAProviderOptions cuda_options;
        //     cuda_options.device_id = 0;
        //     cuda_options.arena_extend_strategy = 0;
        //     cuda_options.gpu_mem_limit = SIZE_MAX;
        //     cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
        //     cuda_options.do_copy_in_default_stream = 1;
        //     options.AppendExecutionProvider_CUDA(cuda_options);
        // }
        // catch (const std::exception &e)
        // {
        //     std::cerr << e.what();
        // }
        options.SetIntraOpNumThreads(4);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        detector = new Detector(env, options, det_path);
        classifier = new Classifier(env, options, cls_path);
        recognizer = new Recognizer(env, options, rec_path, characters_path);
    }
    virtual ~OcrxOnnx()
    {
        delete detector;
        delete classifier;
        delete recognizer;
    }

    virtual std::vector<Result> recognizes(const Image &rgb_image, bool rotated = false) const override
    {
        std::vector<Point> boxes;
        std::vector<Image> images;
        std::vector<float> scores;
        std::vector<std::string> texts;
        std::vector<Result> results;
        detector->process(rgb_image, boxes, images, rotated);
        if (rotated)
            classifier->process(images);
        recognizer->process(images, scores, texts);
        for (int index = 0; index < images.size(); ++index)
        {
            if (scores[index] < 0)
                continue;
            Result result;
            result.box[0] = boxes[index * 3];
            result.box[1] = boxes[index * 3 + 1];
            result.box[2] = boxes[index * 3 + 2];
            result.score = scores[index];
            result.text = std::move(texts[index]);
            results.push_back(result);
        }
        return results;
    }
    virtual Result recognize(const Image &rgb_image) const override
    {
        auto score = 0.0f;
        std::string text;
        recognizer->process(rgb_image, score, text);
        return Result{
            {Point{0, 0}, Point{rgb_image.width, 0}, Point{rgb_image.height, 0}},
            score,
            std::move(text),
        };
    }
};

#ifdef _WIN32
#include <Windows.h>
std::wstring utf8_to_wchar(const char *u8str)
{
    int len = MultiByteToWideChar(CP_UTF8, 0, u8str, -1, nullptr, 0);
    if (len == 0)
        return std::wstring();

    std::wstring wstr(len, 0);
    MultiByteToWideChar(CP_UTF8, 0, u8str, -1, &wstr[0], len);
    return wstr;
}
#endif

Ocrx Ocrx::create(const char *det_path, const char *cls_path, const char *rec_path, const char *characters_path)
{
    OcrxOnnx *ocrx = nullptr;
#ifdef _WIN32
    ocrx = new OcrxOnnx(utf8_to_wchar(det_path).c_str(), utf8_to_wchar(cls_path).c_str(), utf8_to_wchar(rec_path).c_str(), utf8_to_wchar(characters_path).c_str());
#else
    ocrx = new OcrxOnnx(det_path, cls_path, rec_path, characters_path);
#endif
    return Ocrx(ocrx);
}
