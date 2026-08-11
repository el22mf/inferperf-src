import onnxruntime as ort
import numpy as np

# Preallocate everything once at module load
batch = 1
seq_len = 128

# Regular token IDs (prefetcher-friendly)
input_ids = np.ones((batch, seq_len), dtype=np.int64)

# Regular attention mask (branch-predictor-friendly)
attention_mask = np.ones((batch, seq_len), dtype=np.int64)

# Token type IDs (preallocated)
token_type_ids = np.zeros((batch, seq_len), dtype=np.int64)

# Create session once
session = ort.InferenceSession(
    "/home/pi/projects/inferperf-src/inferperf/models/minilm-l12.onnx",
    providers=["CPUExecutionProvider"]
)

# Cache input names once
inputs = session.get_inputs()
input_ids_name = inputs[0].name
attention_name = inputs[1].name
token_type_name = inputs[2].name

# Prebuild input dictionary
feed = {
    input_ids_name: input_ids,
    attention_name: attention_mask,
    token_type_name: token_type_ids
}

def run():
    session.run(None, feed)

if __name__ == "__main__":
    run()
