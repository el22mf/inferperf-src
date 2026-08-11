import onnxruntime as ort
import numpy as np
import time

def run():
    # Load ONNX model
    session = ort.InferenceSession(
        "/home/pi/projects/inferperf-src/inferperf/models/mobilenetv2-7.onnx",
        providers=["CPUExecutionProvider"]
    )

    # Get model input name
    input_name = session.get_inputs()[0].name

    # Synthetic input: 1x3x224x224 float32
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)

    # Timed inference
    start = time.time()
    session.run(None, {input_name: x})
    end = time.time()

    ## Optional printing of inference latency
    #latency_ms = (end - start) * 1000
    #print(f"mobilenetv2 inference latency: {latency_ms:.3f} ms")

if __name__ == "__main__":
    run()
