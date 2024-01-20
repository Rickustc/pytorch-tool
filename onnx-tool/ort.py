import onnxruntime
import numpy as np
import onnx
model_path = "/home/data_share/data_share/ModelS/efficient-b0_simplify.onnx"
onnx_model = onnx.load(model_path)
ort_inputs = {}
img=np.random.rand(1,3,224,224).astype(np.float32)
node_name = []
for node in onnx_model.graph.node:
    for output in node.output:
        # a = onnx.ValueInfoProto(name=output)
        onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])
        node_name.append(node.name)

ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(),providers=['CPUExecutionProvider'])
# ort_session = onnxruntime.InferenceSession(model_path,providers=['CPUExecutionProvider'])

for i, input_ele in enumerate(ort_session.get_inputs()):
    print(i,input_ele.name)
    ort_inputs[input_ele.name] = img
outputs = [x.name for x in ort_session.get_outputs()]
print('outputs num=',len(outputs))
ort_outs = ort_session.run(outputs, ort_inputs)
print('len=',len(ort_outs))
print(ort_outs[0].shape)





import onnxruntime
import numpy as np
import onnx
model_path = "/home/data_share/data_share/ModelS/efficient-b0_simplify.onnx"
onnx_model = onnx.load(model_path)
ort_inputs = {}
# ort_session = onnxruntime.InferenceSession(onnx_model.SerializeToString(),providers=['CPUExecutionProvider'])
ort_session = onnxruntime.InferenceSession(model_path,providers=['CPUExecutionProvider'])

for i, input_ele in enumerate(ort_session.get_inputs()):
    print(i,input_ele.name)
    ort_inputs[input_ele.name] = np.random.rand(1,3,224,224).astype(np.float32)
outputs = [x.name for x in ort_session.get_outputs()]
print('outputs num=',len(outputs))
ort_outs = ort_session.run(outputs, ort_inputs)
print('len=',len(ort_outs))
print(ort_outs[0].shape)

import numpy as np
vec1 = np.array([1, 2, 3, 4])
vec2 = np.array([5, 6, 7, 8])

cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(cos_sim)


import torch
import time
vec1 = torch.FloatTensor([1, 2, 3, 4])
vec2 = torch.FloatTensor([5, 6, 7, 8])
vec1=vec1.flatten()
vec2=vec2.flatten()
t1 = time.time()
cosine_similarity = torch.cosine_similarity(vec1, vec2, dim=0)
print(cosine_similarity)
t2 = time.time()
print(t2-t1)

