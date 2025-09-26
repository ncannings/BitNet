#include "ggml-bitnet.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>

namespace {

static inline size_t row_byte_stride(int columns) {
    const size_t bits_per_row = static_cast<size_t>(columns);
    return (bits_per_row + 7) / 8;
}

static inline bool check_io(float * output, const float * input, const ternary_multiplane_tensor * weight) {
    return output != nullptr && input != nullptr && weight != nullptr;
}

static inline bool check_masks(const ternary_multiplane_tensor * tensor) {
    for (int plane = 0; plane < 3; ++plane) {
        if (tensor->pos_masks[plane] == nullptr || tensor->neg_masks[plane] == nullptr) {
            return false;
        }
    }
    return tensor->group_scales != nullptr || tensor->n_groups == 0;
}

} // namespace

extern "C" void bitnet_multiplane_gemv(
    float * output,
    const float * input,
    const ternary_multiplane_tensor * weight,
    int M,
    int N) {
    if (!check_io(output, input, weight) || !check_masks(weight)) {
        return;
    }

    std::fill_n(output, static_cast<size_t>(M), 0.0f);

    for (int plane = 0; plane < 3; ++plane) {
        const float plane_scale = weight->plane_scales[plane];
        bitnet_gemv_ternary_plane(
            output,
            input,
            weight->pos_masks[plane],
            weight->neg_masks[plane],
            plane_scale,
            M,
            N);
    }

    if (weight->group_scales == nullptr || weight->n_groups <= 0) {
        return;
    }

    const int32_t group_size = weight->group_size;
    for (int32_t group = 0; group < weight->n_groups; ++group) {
        const int32_t start = group * group_size;
        const int32_t end = std::min(start + group_size, weight->n_rows);
        const float scale = weight->group_scales[group];
        for (int32_t row = start; row < end && row < M; ++row) {
            output[row] *= scale;
        }
    }
}

extern "C" void bitnet_gemv_ternary_plane(
    float * output,
    const float * input,
    const uint8_t * pos_mask,
    const uint8_t * neg_mask,
    float scale,
    int M,
    int N) {
    if (output == nullptr || input == nullptr || pos_mask == nullptr || neg_mask == nullptr) {
        return;
    }

    const size_t stride = row_byte_stride(N);
    for (int row = 0; row < M; ++row) {
        const uint8_t * pos_row = pos_mask + stride * static_cast<size_t>(row);
        const uint8_t * neg_row = neg_mask + stride * static_cast<size_t>(row);

        float acc = 0.0f;
        int column = 0;
        for (size_t byte = 0; byte < stride; ++byte) {
            const uint8_t pos_byte = pos_row[byte];
            const uint8_t neg_byte = neg_row[byte];

            for (int bit = 0; bit < 8 && column < N; ++bit, ++column) {
                const uint8_t mask = static_cast<uint8_t>(1u << bit);
                if ((pos_byte & mask) != 0) {
                    acc += input[column];
                }
                if ((neg_byte & mask) != 0) {
                    acc -= input[column];
                }
            }
        }

        output[row] += acc * scale;
    }
}

extern "C" ternary_multiplane_tensor * bitnet_load_multiplane_tensor(FILE * file) {
    if (file == nullptr) {
        return nullptr;
    }

    ternary_multiplane_tensor * tensor = new ternary_multiplane_tensor{};
    tensor->group_scales = nullptr;
    for (int plane = 0; plane < 3; ++plane) {
        tensor->pos_masks[plane] = nullptr;
        tensor->neg_masks[plane] = nullptr;
        tensor->plane_scales[plane] = 0.0f;
    }

    auto cleanup = [&tensor]() {
        bitnet_free_multiplane_tensor(tensor);
        return static_cast<ternary_multiplane_tensor *>(nullptr);
    };

    if (std::fread(&tensor->n_rows, sizeof(int32_t), 1, file) != 1 ||
        std::fread(&tensor->n_cols, sizeof(int32_t), 1, file) != 1 ||
        std::fread(&tensor->group_size, sizeof(int32_t), 1, file) != 1 ||
        std::fread(&tensor->n_groups, sizeof(int32_t), 1, file) != 1) {
        return cleanup();
    }

    if (std::fread(tensor->plane_scales, sizeof(float), 3, file) != 3) {
        return cleanup();
    }

    if (tensor->n_groups < 0) {
        return cleanup();
    }

    if (tensor->n_groups > 0) {
        tensor->group_scales = new float[tensor->n_groups];
        if (std::fread(tensor->group_scales, sizeof(float), tensor->n_groups, file) != static_cast<size_t>(tensor->n_groups)) {
            return cleanup();
        }
    }

    const size_t packed_size = row_byte_stride(tensor->n_cols) * static_cast<size_t>(tensor->n_rows);

    for (int plane = 0; plane < 3; ++plane) {
        tensor->pos_masks[plane] = new uint8_t[packed_size];
        tensor->neg_masks[plane] = new uint8_t[packed_size];

        if (std::fread(tensor->pos_masks[plane], sizeof(uint8_t), packed_size, file) != packed_size ||
            std::fread(tensor->neg_masks[plane], sizeof(uint8_t), packed_size, file) != packed_size) {
            return cleanup();
        }
    }

    return tensor;
}

extern "C" void bitnet_free_multiplane_tensor(ternary_multiplane_tensor * tensor) {
    if (tensor == nullptr) {
        return;
    }

    for (int plane = 0; plane < 3; ++plane) {
        delete[] tensor->pos_masks[plane];
        delete[] tensor->neg_masks[plane];
        tensor->pos_masks[plane] = nullptr;
        tensor->neg_masks[plane] = nullptr;
    }

    delete[] tensor->group_scales;
    tensor->group_scales = nullptr;

    delete tensor;
}

