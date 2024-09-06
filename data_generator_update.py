import numpy as np
import miditoolkit
from chorder import Dechorder

class Preprocessing_update:
    def __init__(self, midi_path):
        self.mid = miditoolkit.midi.parser.MidiFile(midi_path)
        self.tick_per_beat = self.mid.ticks_per_beat

    def all_instrument_notes(self):
        notes = []
        for inst in self.mid.instruments:
            notes.extend(inst.notes)
        return notes

    def onset2syncNotes(self):
        notes = self.all_instrument_notes()
        onset_notes = {}  # {onset:melody note}
        for note in notes:
            if note.start not in onset_notes:
                onset_notes[note.start] = [note]
            else:
                onset_notes[note.start].append(note)
        onset_notes = sorted(onset_notes.items(), key=lambda x: x[0])
        return onset_notes

    def melody(self):
        melody = []
        onset_notes = self.onset2syncNotes()

        for tup in onset_notes:
            pitches = []
            mels = tup[1]
            for note in mels:
                pitches.append(note.pitch)
            mel = mels[pitches.index(max(pitches))]
            melody.append(mel)
        return melody

    def rhythm(self):
        melody = self.melody()
        rhythm_of_melody_in_beats = []

        for i in range(len(melody)-1):
            rhy = (melody[i + 1].start - melody[i].start) / self.tick_per_beat
            rhythm_of_melody_in_beats.append(rhy)

        # approximate the rhythm of last note into the most similar rhythm among other notes
        unique_rhythms = np.array(list(set(rhythm_of_melody_in_beats)))
        last_rhythm = (melody[-1].end - melody[-1].start) / self.tick_per_beat
        temp = list(abs(unique_rhythms - last_rhythm))
        rhythm_of_melody_in_beats.append(unique_rhythms[temp.index(min(temp))])

        # remove trills rhythm which is almost random (here, remove beats of more than 4 digits)
        for r in rhythm_of_melody_in_beats:
            if len(str(r)) >= 6:
                rhythm_of_melody_in_beats.remove(r)

        return rhythm_of_melody_in_beats

    def chord(self):
        num2tone = {0:'C',1:'C#',2:'D',3:'Eb',4:'E',5:'F',6:'F#',7:'G',8:'G#',9:'A',10:'Bb',11:'B', None:'None'}

        def chord_representation(chorder_inst):
            return "".join([num2tone[chorder_inst.root_pc], str(chorder_inst.quality), "/", num2tone[chorder_inst.bass_pc]])

        return list(map(lambda c: chord_representation(c), Dechorder.dechord(self.mid)))

    def melody_bigram(self):
        melody = self.melody()
        melody = list(map(lambda x: x.pitch, melody))
        bigram = []
        for i in range(len(melody)-1):
            bigram.append((melody[i],melody[i+1]))
        return bigram

    def rhythm_bigram(self):
        rhythm = self.rhythm()
        bigram = []
        for i in range(len(rhythm)-1):
            bigram.append((rhythm[i],rhythm[i+1]))
        return bigram

    def chord_bigram(self):
        chord = self.chord()
        bigram = []
        for i in range(len(chord)-1):
            bigram.append((chord[i], chord[i+1]))
        return bigram


