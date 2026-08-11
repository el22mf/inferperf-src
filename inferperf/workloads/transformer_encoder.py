import onnxruntime as ort
import numpy as np

def run():
    session = ort.InferenceSession(
        "/home/pi/projects/inferperf-src/inferperf/models/minilm-l12.onnx",
        providers=["CPUExecutionProvider"]
    )

    inputs = session.get_inputs()
    input_ids_name = inputs[0].name
    attention_name = inputs[1].name
    token_type_name = inputs[2].name

    batch = 4
    seq_len = 512 

    # Irregular token IDs
    input_ids = np.random.randint(0, 30000, size=(batch, seq_len)).astype(np.int64)

    # Irregular attention mask
    attention_mask = np.random.randint(0, 2, size=(batch, seq_len)).astype(np.int64)

    # Token type IDs
    token_type_ids = np.zeros((batch, seq_len), dtype=np.int64)

    session.run(
        None,
        {
            input_ids_name: input_ids,
            attention_name: attention_mask,
            token_type_name: token_type_ids
        }
    )

if __name__ == "__main__":
    run()
