from keramuse.AnnMidiProcessor import AnnMidiProcessor
import matplotlib.pyplot as plt
from parse_args import parse_args

args = parse_args()

amp = AnnMidiProcessor(track=args.track, sequence_length=args.sequence_length)
amp.load_midi(max_midis=args.max_midis, music_dir=args.music_dir)
input_notes, output_notes = amp.construct_sequences()
print(input_notes)
print(output_notes)
amp.make_model(units=args.units, rate=args.rate, activation=args.activation, loss=args.loss, opt=args.opt)
history = amp.train_model(input_notes, output_notes, batch_size=args.batch_size, nb_epoch=args.nb_epoch)
plt.plot(history.history['acc'], label='train')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
# amp.save_model()
