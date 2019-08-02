from music21 import converter, instrument, note, chord, midi, stream
import glob
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout, Activation


class AnnMidiProcessor:
    def __init__(self, track=0, sequence_length=16):
        self.notes = []
        self.durations = []
        self.velocities = []
        # There are multiple tracks in the MIDI file, so we'll use the first one
        self.track = track
        self.pitches = None
        self.sequence_length = sequence_length
        self.note_dict = dict()
        self.vocab_length = 0
        self.model = None
        self.fractions = [1/256, 3/512, 7/1024, 15/2048,
                          1/128, 3/256, 7/512, 15/1024,
                          1/64, 3/128, 7/256, 15/512,
                          1/32, 3/64, 7/128, 15/256,
                          1/16, 3/32, 7/64, 15/128,
                          1/8, 3/16, 7/32, 15/64,
                          1/4, 3/8, 7/16, 15/32,
                          1/2, 3/4, 7/8, 15/16,
                          1, 3/2, 7/4, 15/8,
                          2, 3, 7/2, 15/4,
                          4, 6, 7, 15/2,
                          8, 12, 14, 15]
        print('ANN music generation utility')
        print("Track: {}".format(self.track))
        print("Length of a sequence: {}".format(self.sequence_length))

    def load_midi(self, max_midis=0, music_dir='music'):
        loaded = 0
        for i, file in enumerate(glob.glob("{}/*.mid".format(music_dir))):
            try:
                midi = converter.parse(file)
                midi = midi[self.track]
                notes_to_parse = None
                # Parse the midi file by the notes it contains
                notes_to_parse = midi.flat.notesAndRests
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        self.notes.append(str(element.pitch))
                        self.durations.append(element.quarterLength)
                        self.velocities.append(element.volume.velocityScalar if element.volume.velocityScalar is not None else 0)
                        # print("Note: {}, Velocity: {}, Duration: {}".format(str(element.pitch),
                        #                                                     element.volume.velocityScalar,
                        #                                                     element.quarterLength))
                    elif isinstance(element, chord.Chord):
                        # get's the normal order (numerical representation) of the chord
                        self.notes.append('.'.join(str(n) for n in element.normalOrder))
                        self.durations.append(element.quarterLength)
                        self.velocities.append(element.volume.velocityScalar if element.volume.velocityScalar is not None else 0)
                        # print("Chord: {}, Velocity: {}, Duration: {}".format(
                        #     '.'.join(str(n) for n in element.normalOrder),
                        #     element.volume.velocityScalar, element.quarterLength))
                    else:
                        self.notes.append('R')
                        self.durations.append(element.quarterLength)
                        self.velocities.append(0)
                        # print("Rest. Velocity: {}, Duration: {}".format(0, element.quarterLength))

                print("Song {} {} Loaded".format(i + 1, file.__str__()))
                loaded += 1
                if 0 < max_midis <= loaded:
                    break
            except Exception:
                print("{}: broken song format, message: {} ".format(file.__str__(), Exception))

        print("DONE LOADING SONGS. {} songs are correctly loaded".format(loaded))
        # Get all pitch names
        self.pitches = sorted(set(item for item in self.notes))
        self.fractions = sorted(set(item for item in self.durations)) # update of possible fractions
        print("Number of pitches: {}".format(len(self.pitches)))
        # print(self.pitches)
        print("Number of notes: {}. Number of durations: {}. Number of velocities: {}.".format(len(self.notes), len(self.durations), len(self.velocities)))
        # print(self.notes)
        # print(self.durations)
        # print(self.velocities)

    def construct_sequences(self):
        """
        Now we must get these notes in a usable form for our LSTM.
        Let's construct sequences that can be grouped together to predict the next note in groups of 10 notes.
        Let's use One Hot Encoding for each of the notes and create an array as such of sequences.
        :return:
        """
        # Let's first assign an index to each of the possible notes
        for i, note in enumerate(self.pitches):
            self.note_dict[note] = i
        # Now let's construct sequences.
        # Taking each note and encoding it as a numpy array with a 1 in the position of the note it has
        number_notes = len(self.notes)
        self.vocab_length = len(self.pitches)
        # Lets make a numpy array with the number of training examples, sequence length,
        # and the length of the one-hot-encoding
        num_training = number_notes - self.sequence_length
        print("Number of notes: {};\tLength of a sequence: {};\tNumber of training examples: {}".
              format(number_notes, self.sequence_length, num_training))
        # input_notes = np.zeros((num_training, self.sequence_length, self.vocab_length))
        # output_notes = np.zeros((num_training, self.vocab_length))
        # for i in range(0, num_training):
        #     # Here, i is the training example, j is the note in the sequence for a specific training example
        #     input_sequence = self.notes[i: i + self.sequence_length]
        #     output_note = self.notes[i + self.sequence_length]
        #     for j, note in enumerate(input_sequence):
        #         input_notes[i][j][self.note_dict[note]] = self.durations[self.note_dict[note]]
        #     output_notes[i][self.note_dict[output_note]] = self.durations[self.note_dict[output_note]]
        input_notes = np.zeros((num_training, self.sequence_length, 3))
        output_notes = np.zeros((num_training, 3))
        for i in range(0, num_training):
            # Here, i is the training example, j is the note in the sequence for a specific training example
            input_sequence_of_notes = self.notes[i: i + self.sequence_length]
            input_sequence_of_durations = self.durations[i: i + self.sequence_length]
            input_sequence_of_velocities = self.velocities[i: i + self.sequence_length]
            output_note = self.notes[i + self.sequence_length]
            output_duration = self.durations[i + self.sequence_length]
            output_velocity = self.velocities[i + self.sequence_length]
            for j, note in enumerate(input_sequence_of_notes):
                input_notes[i][j][0] = self.note_dict[note] / (self.vocab_length - 1)
                input_notes[i][j][1] = input_sequence_of_durations[j] / max(self.durations)
                input_notes[i][j][2] = input_sequence_of_velocities[j]
            output_notes[i][0] = self.note_dict[output_note] / (self.vocab_length - 1)
            output_notes[i][1] = output_duration / max(self.durations)
            output_notes[i][2] = output_velocity

        return input_notes, output_notes

    def make_model(self, units=256, rate=0.2, activation='sigmoid',
                   loss='mean_squared_error', metrics=['acc'],
                   opt='rmsprop'):
        self.model = Sequential()
        # self.model.add(LSTM(units, return_sequences=True, input_shape=(self.sequence_length, self.vocab_length)))
        self.model.add(Dropout(rate))
        self.model.add(LSTM(units, return_sequences=True, input_shape=(self.sequence_length, 3)))
        self.model.add(LSTM(units, return_sequences=False))
        self.model.add(Dropout(rate))
        # self.model.add(Dense(self.vocab_length))
        self.model.add(Dense(3))
        self.model.add(Activation(activation))
        self.model.compile(loss=loss, optimizer=opt, metrics=metrics)
        print('Model is compiled. Units: {}. Dropout rate: {}. Activation: {}. Loss: {}. Metrics: {}. Opt: {}'.format(
            units, rate, activation, loss, metrics, opt
        ))

    def train_model(self, input_notes, output_notes, batch_size=4, nb_epoch=400):
        return self.model.fit(input_notes, output_notes, batch_size=batch_size, nb_epoch=nb_epoch)

    def generate_midi(self, input_notes, notes_nb=16, destination='rnn_music', instrumentName='Piano', bars=0):
        # Make a dictionary going backwards (with index as key and the note as the value)
        backward_dict = dict()
        for n in self.note_dict.keys():
            index = self.note_dict[n]
            backward_dict[index] = n

        # pick a random sequence from the input as a starting point for the prediction
        n = np.random.randint(0, len(input_notes) - 1)
        sequence = input_notes[n]

        # start_sequence = sequence.reshape(1, self.sequence_length, self.vocab_length)
        start_sequence = sequence.reshape(1, self.sequence_length, 3)
        # print(start_sequence)
        output = []
        # Let's generate a song of notes_nb notes
        for i in range(0, notes_nb):
            new_note = self.model.predict(start_sequence, verbose=0)
            # Get the position with the highest probability
            # index = np.argmax(new_note)
            # encoded_note = np.zeros((self.vocab_length))
            # encoded_note[index] = np.max(new_note)
            # output.append(encoded_note)
            # sequence = start_sequence[0][1:]
            # start_sequence = np.concatenate((sequence, encoded_note.reshape(1, self.vocab_length)))
            # start_sequence = start_sequence.reshape(1, self.sequence_length, self.vocab_length)
            # print(new_note[0])
            output.append(new_note[0])
            sequence = start_sequence[0][1:]
            start_sequence = np.concatenate((sequence, new_note.reshape(1, 3)))
            start_sequence = start_sequence.reshape(1, self.sequence_length, 3)
        # Now output is populated with notes in their string form
        # for element in output:
        #     print(element)

        final_notes = []
        final_durations = []
        final_velocities = []
        for element in output:
            # index = list(element).index(1)
            # index = np.argmax(list(element))
            # final_notes.append(backward_dict[index])
            # final_durations.append(self.get_closest_value(self.fractions, np.max(list(element))))
            final_notes.append(backward_dict[round(element[0] * (self.vocab_length - 1))])
            final_durations.append(self.get_closest_value(arr=self.fractions,
                                                          target=(element[1] * max(self.durations))))
            final_velocities.append(element[2])

        print("File: output/{}.mid: ".format(destination))
        print("\tNotes: {}".format(final_notes))
        print("\tDurations: {}".format(final_durations))
        print("\tVelocities: {}".format(final_velocities))

        offset = 0
        output_notes = []
        total_duration = sum(final_durations)
        midi_instrument = getattr(instrument, instrumentName)
        output_notes.append(midi_instrument())

        # create note and chord objects based on the values generated by the model
        for pattern, duration, velocity in zip(final_notes, final_durations, final_velocities):
            mapped_duration = (4 * bars * duration / total_duration) if (bars > 0) else (duration)
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = midi_instrument()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.quarterLength = mapped_duration
                new_chord.volume.velocityScalar = velocity
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a rest
            elif pattern == 'R':
                new_note = note.Rest()
                new_note.offset = offset
                new_note.quarterLength = mapped_duration
                new_note.storedInstrument = midi_instrument()
                output_notes.append(new_note)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.quarterLength = mapped_duration
                new_note.volume.velocityScalar = velocity
                new_note.storedInstrument = midi_instrument()
                output_notes.append(new_note)

            # increase offset each iteration so that notes do not stack
            offset += mapped_duration

        midi_stream = stream.Stream(output_notes)

        midi_stream.write('midi', fp="output/{}.mid".format(destination))

    def save_model(self, file_name="model/model"):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(file_name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(file_name + ".h5")
        print("ANN model is saved to disk: {} {}".format(file_name + ".json", file_name + ".h5"))

    def load_model(self, file_name="model/model", loss='mean_squared_error',
                   metrics=['acc'], opt='rmsprop'):
        # load json and create model
        json_file = open(file_name + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(file_name + ".h5")
        self.model.compile(loss=loss, optimizer=opt, metrics=metrics)
        print("ANN model is loaded from disk: {} {}".format(file_name + ".json", file_name + ".h5"))

    @staticmethod
    def get_closest_value(arr, target):
        n = len(arr)
        left = 0
        right = n - 1
        mid = 0

        # edge case - last or above all
        if target >= arr[n - 1]:
            return arr[n - 1]
        # edge case - first or below all
        if target <= arr[0]:
            return arr[0]
        # BSearch solution: Time & Space: Log(N)

        while left < right:
            mid = (left + right) // 2  # find the mid
            if target < arr[mid]:
                right = mid
            elif target > arr[mid]:
                left = mid + 1
            else:
                return arr[mid]

        if target < arr[mid]:
            return arr[mid] if target - arr[mid - 1] >= arr[mid] - target else arr[mid - 1]
        else:
            return arr[mid + 1] if target - arr[mid] >= arr[mid + 1] - target else arr[mid]
