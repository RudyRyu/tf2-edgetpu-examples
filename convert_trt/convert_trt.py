from tensorflow.python.compiler.tensorrt import trt_convert as trt

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16,
    max_workspace_size_bytes=8000000000)

converter = trt.TrtGraphConverterV2(
   input_saved_model_dir='checkpoints/LPD_mobilenet_v2_keras_pretrained_v1/saved_model', 
   conversion_params=conversion_params)

converter.convert()
converter.save('checkpoints/LPD_mobilenet_v2_keras_pretrained_v1/trt')
print('Done Converting to TF-TRT FP16')
