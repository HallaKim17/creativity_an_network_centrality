import pretty_midi
import collections
import numpy as np
import math


class Preprocessing:
    def __init__(self, file_name):
        self.file_name = file_name
        self.midi = pretty_midi.PrettyMIDI(self.file_name)

        self.notes = [note for inst in self.midi.instruments for note in inst.notes]
        self.times = sorted(list(set([note.start for note in self.notes])))  # all unique onsets

        self.synchronous_notes = {}
        for time in self.times:
            self.synchronous_notes[time] = []
            for note in self.notes:
                if time == note.start:
                    self.synchronous_notes[time].append(note)
        self.synchronous_notes = list(self.synchronous_notes.values())
        self.synchronous_notes_names = [[pretty_midi.note_number_to_name(note.pitch) for note in sync_notes] for sync_notes in self.synchronous_notes]
        self.synchronous_notes_pitches = [[note.pitch for note in sync_notes] for sync_notes in self.synchronous_notes]

    def melody(self):
        highest_notes = [max(sync_notes) for sync_notes in self.synchronous_notes_pitches]
        return highest_notes

    def melody_name(self):
        note_number = self.melody()
        melody_name = [pretty_midi.note_number_to_name(n) for n in note_number]
        return melody_name

    def bigram(self):
        melody = self.melody()
        bigram = []
        for i in range(len(melody)-1):
            bigram.append((melody[i],melody[i+1]))
        return bigram

    def bigram_name(self):
        bigram = self.bigram()
        bigram_name = [tuple([pretty_midi.note_number_to_name(note) for note in bi]) for bi in bigram]
        return bigram_name


def melody_bigram_plane(midi):
    assert isinstance(midi, Preprocessing)
    bigram = midi.bigram()
    bigram = dict(collections.Counter(bigram))
    plane = np.zeros((88,88))
    for (b1, b2) in bigram:
        plane[b1-21, b2-21] = bigram[(b1,b2)] / sum(bigram.values())
    return plane


def bigram_distance(midi1, midi2):
    midi1_prob_dist = melody_bigram_plane(midi1)
    midi2_prob_dist = melody_bigram_plane(midi2)

    midi1_bigram = dict(collections.Counter(midi1.bigram()))
    midi2_bigram = dict(collections.Counter(midi2.bigram()))
    all_bigrams = list(set(midi1_bigram) | set(midi2_bigram))

    error_sum = 0
    for (b1, b2) in all_bigrams:
        error = (midi1_prob_dist[b1-21, b2-21] - midi2_prob_dist[b1-21, b2-21]) ** 2
        error_sum += error
    distance = math.sqrt(error_sum)
    return distance


