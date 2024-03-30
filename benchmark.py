import numpy as np
from deepsparse.engine import benchmark_model


def benchmark(model, sample_inputs, batch_size):
    results = benchmark_model(
        model,
        [sample_inputs[:batch_size]],
        batch_size=batch_size,
        num_cores=4,
        num_iterations=50,
        num_warmup_iterations=5,
    )
    print(f'{model}, {batch_size=}: {results=}')


if __name__ == "__main__":
    sample_inputs = np.random.randn(32, 3, 32, 32).astype(np.float32)
    benchmark('checkpoints/baseline_resnet.onnx', sample_inputs, 1)
    benchmark('checkpoints/baseline_resnet.onnx', sample_inputs, 32) 
    benchmark('pruned_quantized_resnet.onnx', sample_inputs, 1)
    benchmark('pruned_quantized_resnet.onnx', sample_inputs, 32)
