from keramuse.AnnMidiProcessor import AnnMidiProcessor
import matplotlib.pyplot as plt
from parse_args import parse_args


def plot_history(history, metric='loss', title='Model Loss', xlabel='epoch', ylabel='loss'):
    plt.plot(history.history[metric])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


args = parse_args()

amp = AnnMidiProcessor(track=args.track, sequence_length=args.sequence_length)
amp.load_midi(max_midis=args.max_midis, music_dir=args.music_dir)
amp.construct_sequences()
durations_history, offsets_history, notes_history, velocities_history = amp.train(units=args.units, rate=args.rate,
                                                                                  activation=args.activation,
                                                                                  loss=args.loss, opt=args.opt,
                                                                                  batch_size=args.batch_size,
                                                                                  nb_epoch=args.nb_epoch)

if not args.plot:
    plot_history(notes_history, title='Notes Model Loss')
    plot_history(notes_history, metric='acc', title='Notes Model Accuracy', ylabel='accuracy')
    plot_history(durations_history, title='Durations Model Loss')
    plot_history(durations_history, metric='acc', title='Durations Model Accuracy', ylabel='accuracy')
    plot_history(offsets_history, title='Offsets Model Loss')
    plot_history(offsets_history, metric='acc', title='Offsets Model Accuracy', ylabel='accuracy')
    plot_history(velocities_history, title='Velocities Model Loss')
    plot_history(velocities_history, metric='acc', title='Velocities Model Accuracy', ylabel='accuracy')

amp.save_model()
