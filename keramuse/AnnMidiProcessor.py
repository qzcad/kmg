from music21 import converter, instrument, note, chord, midi, stream
import glob
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout, Activation
from tqdm import tqdm
import pickle
import os


class AnnMidiProcessor:
    def __init__(self, track=0, sequence_length=16):
        self.notes = []
        self.durations = []
        self.velocities = []
        self.offsets = []
        self.positions = []
        # There are multiple tracks in the MIDI file, so we'll use the first one
        self.track = track
        self.sequence_length = sequence_length

        self.notes_model = None
        self.durations_model = None
        self.offsets_model = None
        self.velocities_model = None

        print('ANN music generation utility')
        print("Track: {}".format(self.track))
        print("Length of a sequence: {}".format(self.sequence_length))

    def load_midi(self, max_midis=0, music_dir='music'):
        loaded = 0
        position = 0

        for i, file in tqdm(enumerate(glob.glob("{}/*.mid".format(music_dir)))):
            try:
                midi = converter.parse(file)
                midi = midi[self.track]
                notes_to_parse = None
                # Parse the midi file by the notes it contains
                notes_to_parse = midi.flat.notesAndRests
                if len(notes_to_parse) > 0:
                    # append position of a new melody in the notes list
                    self.positions.append(position)
                total_offset = 0
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        self.notes.append(str(element.pitch))
                        self.durations.append(element.quarterLength)
                        self.velocities.append(element.volume.velocity if element.volume.velocity is not None else 0)
                        self.offsets.append(element.offset - total_offset)
                        total_offset += element.offset - total_offset
                        # print("Note: {}, Velocity: {}, Duration: {}, Offset: {}".format(str(element.pitch),
                        #                                                                 element.volume.velocity,
                        #                                                                 element.quarterLength,
                        #                                                                 element.offset))
                    elif isinstance(element, chord.Chord):
                        # get's the normal order (numerical representation) of the chord
                        self.notes.append('.'.join(str(n) for n in element.normalOrder))
                        self.durations.append(element.quarterLength)
                        self.velocities.append(element.volume.velocity if element.volume.velocity is not None else 0)
                        self.offsets.append(element.offset - total_offset)
                        total_offset += element.offset - total_offset
                        # print("Chord: {}, Velocity: {}, Duration: {}, Offset: {}".format(
                        #     '.'.join(str(n) for n in element.normalOrder),
                        #     element.volume.velocity, element.quarterLength, element.offset))
                    else:
                        self.notes.append('R')
                        self.durations.append(element.quarterLength)
                        self.velocities.append(0)
                        self.offsets.append(element.offset - total_offset)
                        total_offset += element.offset - total_offset
                        # print("Rest. Velocity: {}, Duration: {}, Offset: {}".format(0, element.quarterLength, element.offset))

                    position += 1

                print("Song {} {} Loaded".format(i + 1, file.__str__()))
                loaded += 1
                if 0 < max_midis <= loaded:
                    break
            except Exception:
                print("{}: broken song format, message: {} ".format(file.__str__(), Exception))

        print("DONE LOADING SONGS. {} songs are correctly loaded".format(loaded))
        print("Notes len is {}, Durations len is {}, Offsets len is {}, Velocities len is {}".format(len(self.notes),
                                                                                                     len(
                                                                                                         self.durations),
                                                                                                     len(self.offsets),
                                                                                                     len(
                                                                                                         self.velocities)))
        print(self.notes)

    @staticmethod
    def __dict_from_list__(l):
        unique = sorted(set(l))
        return {unique[i]: i for i in range(0, len(unique))}

    @staticmethod
    def __backward_dict_from_list__(l):
        unique = sorted(set(l))
        return {i: unique[i] for i in range(0, len(unique))}

    @staticmethod
    def __construct_sequences__(
            values,
            sequence_length
    ):
        number_of_values = len(values)
        values_dict = AnnMidiProcessor.__dict_from_list__(values)
        number_training = number_of_values - sequence_length
        input = np.zeros((number_training, sequence_length, len(values_dict)))
        output = np.zeros((number_training, len(values_dict)))
        for i in range(0, number_training):
            # onehot encoding of pitches
            # Here, i is the training example, j is the value in the sequence for a specific training example
            input_sequence = values[i: i + sequence_length]
            output_value = values[i + sequence_length]

            for j, v in enumerate(input_sequence):
                input[i][j][values_dict[v]] = 1

            output[i][values_dict[output_value]] = 1

        return input, output

    def construct_sequences(self):
        self.input_notes, self.output_notes = self.__construct_sequences__(self.notes, self.sequence_length)
        self.input_durations, self.output_durations = self.__construct_sequences__(self.durations, self.sequence_length)
        self.input_offsets, self.output_offsets = self.__construct_sequences__(self.offsets, self.sequence_length)
        self.input_velocites, self.output_velocities = self.__construct_sequences__(self.velocities,
                                                                                    self.sequence_length)

    @staticmethod
    def __lstm__(
            sequence_length,
            vocab_length,
            units=256,
            rate=0.2,
            activation='softmax',
            loss='categorical_crossentropy',
            metrics=['acc'],
            opt='rmsprop'
    ):
        model = Sequential()
        model.add(LSTM(units, return_sequences=True, input_shape=(sequence_length, vocab_length)))
        model.add(Dropout(rate))
        model.add(LSTM(units, return_sequences=False))
        model.add(Dropout(rate))
        model.add(Dense(vocab_length))
        model.add(Activation(activation))
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        return model

    def train(
            self,
            units=256,
            rate=0.2,
            activation='softmax',
            loss='categorical_crossentropy',
            metrics=['acc'],
            opt='rmsprop',
            batch_size=4,
            nb_epoch=400):
        self.notes_model = self.__lstm__(self.sequence_length,
                                         len(set(self.notes)),
                                         units=units,
                                         rate=rate,
                                         activation=activation,
                                         loss=loss,
                                         metrics=metrics,
                                         opt=opt)
        self.durations_model = self.__lstm__(self.sequence_length,
                                             len(set(self.durations)),
                                             units=units,
                                             rate=rate,
                                             activation=activation,
                                             loss=loss,
                                             metrics=metrics,
                                             opt=opt)
        self.offsets_model = self.__lstm__(self.sequence_length,
                                           len(set(self.offsets)),
                                           units=units,
                                           rate=rate,
                                           activation=activation,
                                           loss=loss,
                                           metrics=metrics,
                                           opt=opt)
        self.velocities_model = self.__lstm__(self.sequence_length,
                                              len(set(self.velocities)),
                                              units=units,
                                              rate=rate,
                                              activation=activation,
                                              loss=loss,
                                              metrics=metrics,
                                              opt=opt)
        notes_history = self.notes_model.fit(self.input_notes, self.output_notes, batch_size=batch_size,
                                             nb_epoch=nb_epoch)
        durations_history = self.durations_model.fit(self.input_durations, self.output_durations, batch_size=batch_size,
                                                     nb_epoch=nb_epoch)
        offsets_history = self.offsets_model.fit(self.input_offsets, self.output_offsets, batch_size=batch_size,
                                                 nb_epoch=nb_epoch)
        velocities_history = self.velocities_model.fit(self.input_velocites, self.output_velocities,
                                                       batch_size=batch_size, nb_epoch=nb_epoch)
        return durations_history, offsets_history, notes_history, velocities_history

    @staticmethod
    def __predict__(notes_nb, model, start_sequence, dict_len, sequence_length):
        output = []
        for i in range(0, notes_nb):
            new_value = model.predict(start_sequence, verbose=0)
            # Get the position with the highest probability
            index = np.argmax(new_value)
            encoded_value = np.zeros(dict_len)
            encoded_value[index] = 1.0
            output.append(encoded_value)
            sequence = start_sequence[0][1:]
            start_sequence = np.concatenate((sequence, encoded_value.reshape(1, dict_len)))
            start_sequence = start_sequence.reshape(1, sequence_length, dict_len)
        return output

    def generate_midi(self, notes_nb=16, destination='rnn_music', instrumentName='Piano', bars=0):
        # Make dictionaries going backwards (with index as key and the note as the value)
        notes_backward_dict = self.__backward_dict_from_list__(self.notes)
        durations_backward_dict = self.__backward_dict_from_list__(self.durations)
        offsets_backward_dict = self.__backward_dict_from_list__(self.offsets)
        velocities_backward_dict = self.__backward_dict_from_list__(self.velocities)

        notes_dict_len = len(notes_backward_dict)
        durations_dict_len = len(durations_backward_dict)
        offsets_dict_len = len(offsets_backward_dict)
        velocities_dict_len = len(velocities_backward_dict)

        # pick random sequences from the input as a starting point for the prediction
        n = np.random.randint(0, len(self.input_notes) - 1)
        notes_sequence = self.input_notes[n]
        durations_sequence = self.input_durations[n]
        offsets_sequence = self.input_offsets[n]
        velocities_sequence = self.input_velocites[n]

        notes_start_sequence = notes_sequence.reshape(1, self.sequence_length, notes_dict_len)
        durations_start_sequence = durations_sequence.reshape(1, self.sequence_length, durations_dict_len)
        offsets_start_sequence = offsets_sequence.reshape(1, self.sequence_length, offsets_dict_len)
        velocities_start_sequence = velocities_sequence.reshape(1, self.sequence_length, velocities_dict_len)
        # print(start_sequence)
        notes_output = self.__predict__(notes_nb, self.notes_model, notes_start_sequence, notes_dict_len, self.sequence_length)
        durations_output = self.__predict__(notes_nb, self.durations_model, durations_start_sequence, durations_dict_len, self.sequence_length)
        offsets_output = self.__predict__(notes_nb, self.offsets_model, offsets_start_sequence, offsets_dict_len, self.sequence_length)
        velocities_output = self.__predict__(notes_nb, self.velocities_model, velocities_start_sequence, velocities_dict_len, self.sequence_length)

        final_notes = []
        final_durations = []
        final_velocities = []
        final_offsets = []
        for n, d, o, v in zip(notes_output, durations_output, offsets_output, velocities_output):
            final_notes.append(notes_backward_dict[list(n).index(1)])
            final_durations.append(durations_backward_dict[list(d).index(1)])
            final_offsets.append(offsets_backward_dict[list(o).index(1)])
            final_velocities.append(velocities_backward_dict[list(v).index(1)])

        print("File: output/{}.mid: ".format(destination))
        print("\tNotes: {}".format(final_notes))
        print("\tDurations: {}".format(final_durations))
        print("\tOffsets: {}".format(final_offsets))
        print("\tVelocities: {}".format(final_velocities))

        total_offset = 0
        output_notes = []
        total_duration = sum(final_durations)
        midi_instrument = getattr(instrument, instrumentName)
        output_notes.append(midi_instrument())

        # create note and chord objects based on the values generated by the model
        for pattern, duration, offset, velocity in zip(final_notes, final_durations, final_offsets, final_velocities):
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
                new_chord.volume.velocity = velocity
                new_chord.offset = total_offset
                output_notes.append(new_chord)
            # pattern is a rest
            elif pattern == 'R':
                new_note = note.Rest()
                new_note.offset = total_offset
                new_note.quarterLength = mapped_duration
                new_note.storedInstrument = midi_instrument()
                output_notes.append(new_note)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = total_offset
                new_note.quarterLength = mapped_duration
                new_note.volume.velocity = velocity
                new_note.storedInstrument = midi_instrument()
                output_notes.append(new_note)

            # increase offset each iteration so that notes do not stack
            total_offset += offset

        midi_stream = stream.Stream(output_notes)

        midi_stream.write('midi', fp="output/{}.mid".format(destination))

    @staticmethod
    def __save_model__(model, name_pattern):
        # serialize model to JSON
        model_json = model.to_json()
        with open(name_pattern + ".json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(name_pattern + ".h5")

    def save_model(self, dir_name="model"):
        self.__save_model__(self.notes_model, os.path.join(dir_name, 'notes_model'))
        self.__save_model__(self.durations_model, os.path.join(dir_name, 'durations_model'))
        self.__save_model__(self.offsets_model, os.path.join(dir_name, 'offsets_model'))
        self.__save_model__(self.velocities_model, os.path.join(dir_name, 'velocities_model'))
        self.save_obj(self.notes, 'notes')
        self.save_obj(self.durations, 'durations')
        self.save_obj(self.offsets, 'offsets')
        self.save_obj(self.velocities, 'velocities')

    @staticmethod
    def __load_model__(name_pattern, loss='mean_squared_error', metrics=['acc'], opt='rmsprop'):
        # load json and create model
        json_file = open(name_pattern + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(name_pattern + ".h5")
        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        return model

    def load_model(self, dir_name="model", loss='mean_squared_error',
                   metrics=['acc'], opt='rmsprop'):
        self.notes_model = self.__load_model__(os.path.join(dir_name, 'notes_model'), loss, metrics, opt)
        self.durations_model = self.__load_model__(os.path.join(dir_name, 'durations_model'), loss, metrics, opt)
        self.offsets_model = self.__load_model__(os.path.join(dir_name, 'offsets_model'), loss, metrics, opt)
        self.velocities_model = self.__load_model__(os.path.join(dir_name, 'velocities_model'), loss, metrics, opt)

        self.notes = self.load_obj('notes')
        self.durations = self.load_obj('durations')
        self.offsets = self.load_obj('offsets')
        print(self.offsets)
        self.velocities = self.load_obj('velocities')

        self.construct_sequences()

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

    @staticmethod
    def save_obj(obj, name):
        with open('model/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(name):
        with open('model/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
