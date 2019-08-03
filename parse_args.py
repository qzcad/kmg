import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ANN music generation utility')
    parser.add_argument('--music_dir', type=str, nargs='?', default='music',
                        help='Absolute or relative path to a train set. Type: str. Default: music.')
    parser.add_argument('--instrument', type=str, nargs='?', default='Piano',
                        help='An instrument of generated midi. '
                             'See http://web.mit.edu/music21/doc/moduleReference/moduleInstrument.html. '
                             'Type: str. Default: Piano.')
    parser.add_argument('--notes', type=int, nargs='?', default=16,
                        help='A number of notes to generate at one midi-file. Type: int. Default: 16.')
    parser.add_argument('--bars', type=int, nargs='?', default=0,
                        help='A number of bars. If bars > 0 than durations is mapped on 4*bars interval.'
                             ' Type: int. Default: 0.')
    parser.add_argument('--track', type=int, nargs='?', default=0,
                        help='The number of a track in a midi-file. Type: int. Default: 0.')
    parser.add_argument('--sequence_length', type=int, nargs='?', default=16,
                        help='The length of a sequence. Type: int. Default: 16.')
    parser.add_argument('--max_midis', type=int, nargs='?', default=0,
                        help='The limit of the count of processed files. '
                             'If less or equal 0 then the count of processed files is unlimited. '
                             'Type: int. Default: 0.')
    parser.add_argument('--units', type=int, nargs='?', default=256,
                        help='Positive integer, dimensionality of the output space in LSTM layers.'
                             'Type: int. Default: 256.')
    parser.add_argument('--rate', type=float, nargs='?', default=0.2,
                        help='Fraction of the input units to drop. Type: float between 0 and 1. Default: 0.2.')
    parser.add_argument('--activation', type=str, nargs='?', default='sigmoid',
                        help='An activation function. See https://keras.io/activations/. Type: str. Default: sigmoid.')
    parser.add_argument('--loss', type=str, nargs='?', default='mean_squared_error',
                        help='A loss function. See https://keras.io/losses/. '
                             'Type: str. Default: mean_squared_error')
    parser.add_argument('--opt', type=str, nargs='?', default='rmsprop',
                        help='An optimizer. See https://keras.io/optimizers/. Type: str. Default: rmsprop.')
    parser.add_argument('--batch_size', type=int, nargs='?', default=4,
                        help='A number of training examples in one forward/backward pass. Type: int. Default: 4.')
    parser.add_argument('--nb_epoch', type=int, nargs='?', default=400,
                        help='A number of epochs to train the model. Type: int. Default: 400.')
    parser.add_argument('--destination', type=str, nargs='?', default='rnn_music',
                        help='A name of a file to store generated midi. Type: str. Default: rnn_music.')
    parser.add_argument('--midis', type=int, nargs='?', default=1,
                        help='A number of generated midi-files. Type: int. Default: 1')
    parser.add_argument("-p", "--plot", action="store_true", help="Plot accuracy and loss of the model.")
    return parser.parse_args()
