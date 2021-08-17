
from model_utils.export_tflite_graph import export_tflite_graph
import train.config
config = train.config.config

model_name = ''
checkpoint_name = ''

# 1.create saved model
meta_info_path = './checkpoints/{}'.format(model_name)
export_tflite_graph(meta_info_path+'/meta_info.config', meta_info_path,
                    checkpoint_name=checkpoint_name)

# 2. convert to tflite
 

                

