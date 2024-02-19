import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
import torch
import torch.nn.functional as F
import tensorflow as tf
from model import Mamba, ModelArgs
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

@nobuco.converter(F.softplus, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
def softplus(input: torch.Tensor):
    return lambda input: tf.keras.activations.softplus(input)

args = ModelArgs(
    d_model=5,
    n_layer=1,
    vocab_size=50277
)
model = Mamba(args)
model.eval()
export_name = "mamba_minimal_1_layer"
dummy_input = "Test"
input_ids = tokenizer(dummy_input, return_tensors='pt').input_ids

keras_model = nobuco.pytorch_to_keras(
    model,
    args=[input_ids], kwargs=None,
    input_shapes={input_ids: (1, None)}, # Annotate dynamic axes with None
    inputs_channel_order=ChannelOrder.TENSORFLOW,
    outputs_channel_order=ChannelOrder.TENSORFLOW,
    constants_to_variables=False,
    trace_shape=True,
    save_trace_html=True
)

tf.keras.models.save_model(keras_model, f'{export_name}.keras')
tf.keras.models.save_model(keras_model, f'{export_name}.h5')
tf.saved_model.save(keras_model, f'{export_name}')