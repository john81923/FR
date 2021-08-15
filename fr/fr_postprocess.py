import numpy as np
def postprocess_(outputs, **kwargs):
    outputs = np.squeeze(outputs)
    normed_emb = outputs / np.linalg.norm(outputs)
    return normed_emb