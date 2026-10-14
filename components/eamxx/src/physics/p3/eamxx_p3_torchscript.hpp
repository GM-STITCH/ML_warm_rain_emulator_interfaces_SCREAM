#ifndef SCREAM_P3_TORCHSCRIPT_HPP
#define SCREAM_P3_TORCHSCRIPT_HPP

#include <torch/script.h>

#include <array>
#include <mutex>
#include <string>

namespace scream {

// Host-side, CPU-only TorchScript wrapper used by the P3 autoconversion path.
class P3TorchScriptModel {
public:
  static constexpr int num_levels = 16;
  static constexpr int num_inputs = 11;
  static constexpr int num_outputs = 4;
  using Input = std::array<double, num_levels * num_inputs>;
  using Output = std::array<double, num_levels * num_outputs>;

  explicit P3TorchScriptModel(const std::string& model_path)
    : module(torch::jit::load(model_path, torch::kCPU))
  {
    module.eval();
  }

  Output infer(const Input& values) const
  {
    // Serialize calls and keep LibTorch's own thread pools from racing with
    // the P3 CPU team loop. The caller receives the model values unchanged.
    std::lock_guard<std::mutex> lock(mutex);
    torch::InferenceMode inference_mode;
    const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto input = torch::from_blob(const_cast<double*>(values.data()),
                                  {num_inputs, num_levels}, options).clone();
    auto output = module.forward({input}).toTensor().to(torch::kFloat64).contiguous();
    TORCH_CHECK(output.dim() == 2 && output.size(0) == num_outputs &&
                output.size(1) == num_levels,
                "P3 TorchScript model must return a [4, 16] tensor; got ", output.sizes());

    Output result{};
    const auto accessor = output.accessor<double, 2>();
    for (int j = 0; j < num_outputs; ++j) {
      for (int k = 0; k < num_levels; ++k) {
        result[k * num_outputs + j] = accessor[j][k];
      }
    }
    return result;
  }

private:
  mutable torch::jit::script::Module module;
  mutable std::mutex mutex;
};

} // namespace scream

#endif // SCREAM_P3_TORCHSCRIPT_HPP
