
import onnx
import onnx_tool
import onnxruntime
import onnx
import numpy as np


input_path = "/home/fiery/work/vendors/intellif/DEngine_edge10_v1.3.0-fuxiao/tyassist/models/O2_models/petr_e1.onnx"


onnx_model = onnx.load(input_path)
op_dict = {}



for node in onnx_model.graph.node:

    assert node.op_type+"Node" in onnx_tool.NODE_REGISTRY.keys() , "not suopported node for onnx-tool"
    if node.op_type in op_dict:
        op_dict[node.op_type] += 1
    else:
        op_dict[node.op_type] = 1

print("all op types: ")
print(op_dict)


print("onnx tool infer: ")
onnx_tool.model_profile(input_path,saveshapesmodel='./shapes.onnx')



print("onnxruntime infer")
# model_path="/home/fiery/work/vendors/intellif/DEngine_edge10_v1.3.0-fuxiao/tyassist/models/onnx/onnx_resnet50/modified_ort_resnet_quant_fx_reference_qdq_shift_scale.onnx"
model_path = "/home/fiery/work/vendors/intellif/DEngine_edge10_v1.3.0-fuxiao/tyassist/models/onnx/onnx_resnet50/resnet50_int8.onnx"
# chehck ort
sess = onnxruntime.InferenceSession(model_path)
# 获取输入名称和形状
input_details = sess.get_inputs()
for input_detail in input_details:
    print(f"Input Name: {input_detail.name}, Input Shape: {input_detail.shape}")
    
    
model_outputs = sess.get_outputs()
output_names = [model_outputs[i].name for i in range(len(model_outputs))]
print(output_names)

model_inputs =sess.get_inputs()
input_names = [model_inputs[i].name for i in range(len(model_inputs))]

input_tensor = np.random.rand(1,3,224,224)
input_tensor  = input_tensor.astype(np.float32)
outputs = sess.run(output_names, {input_names[0]: input_tensor})
print(outputs[0].shape)


