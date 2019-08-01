from keramuse.AnnMidiProcessor import AnnMidiProcessor
import matplotlib.pyplot as plt

amp = AnnMidiProcessor()
amp.load_midi()
input_notes, output_notes = amp.construct_sequences()
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
amp.make_model(activation='softmax')
history = amp.train_model(input_notes, output_notes, batch_size=4)
plt.plot(history.history['acc'], label='softmax')
amp.make_model(activation='softplus')
history = amp.train_model(input_notes, output_notes, batch_size=4)
plt.plot(history.history['acc'], label='softplus')
amp.make_model(activation='sigmoid')
history = amp.train_model(input_notes, output_notes, batch_size=4)
plt.plot(history.history['acc'], label='sigmoid')
amp.make_model(activation='hard_sigmoid')
history = amp.train_model(input_notes, output_notes, batch_size=4)
plt.plot(history.history['acc'], label='hard_sigmoid')
plt.legend()
plt.show()