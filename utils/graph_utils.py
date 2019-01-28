import os
from argparse import ArgumentParser
from tensorflow.python.tools import freeze_graph


if __name__ == '__main__':
    parser = ArgumentParser('Graph freezer and exporter.')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model files.')
    parser.add_argument('--step', type=int, required=True, help='Step to use for exporting.')
    args = parser.parse_args()

    input_graph_path = os.path.join(args.model_dir, 'graph.pbtxt')
    checkpoint_path = os.path.join(args.model_dir, 'model.ckpt-{}'.format(args.step))
    input_saver_def_path = ""
    input_binary = False
    output_node_names = ','.join(
        ['listener/bidirectional_rnn/fw/fw/while/Exit_3',  # fw-c-1
         'listener/bidirectional_rnn/fw/fw/while/Exit_4',  # fw-h-1
         'listener/bidirectional_rnn/fw/fw/while/Exit_5',  # fw-c-2
         'listener/bidirectional_rnn/fw/fw/while/Exit_6',  # fw-h-2
         'listener/bidirectional_rnn/bw/bw/while/Exit_3',  # bw-c-1
         'listener/bidirectional_rnn/bw/bw/while/Exit_4',  # bw-h-1
         'listener/bidirectional_rnn/bw/bw/while/Exit_5',  # bw-c-2
         'listener/bidirectional_rnn/bw/bw/while/Exit_6']  # bw-h-2
    )
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = os.path.join(args.model_dir, 'frozen_graph.pb')
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")
