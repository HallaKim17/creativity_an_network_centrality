import miditoolkit
import numpy as np
# import music21
# from music21 import converter


def rhythm_of_melodies(midi_path):
    mid = miditoolkit.midi.parser.MidiFile(midi_path)
    tick_per_beat = mid.ticks_per_beat
    notes = []
    for inst in mid.instruments:
        notes.extend(inst.notes)
    onset_notes = {} # {onset:melody note}
    for note in notes:
        if note.start not in onset_notes:
            onset_notes[note.start] = [note]
        else:
            onset_notes[note.start].append(note)
    #print(onset_notes)
    onset_notes = sorted(onset_notes.items(), key=lambda x: x[0])
    melody = []
    rhythm_of_melody_in_beats = []
    for tup in onset_notes:
        pitches = []
        mels = tup[1]
        for note in mels:
            pitches.append(note.pitch)
        mel = mels[pitches.index(max(pitches))]
        melody.append(mel)
    for i in range(len(melody)-1):
        rhy = (melody[i+1].start-melody[i].start)/tick_per_beat
        rhythm_of_melody_in_beats.append(rhy)
    # approximate the rhythm of last note into the most similar rhythm among other notes
    unique_rhythms = np.array(list(set(rhythm_of_melody_in_beats)))
    last_rhythm = (melody[-1].end-melody[-1].start)/tick_per_beat
    temp = list(abs(unique_rhythms-last_rhythm))
    rhythm_of_melody_in_beats.append(unique_rhythms[temp.index(min(temp))])
    return melody, rhythm_of_melody_in_beats


midi_path = "../../dataset_git/Dataset/Bach/goldberg_variations_988_30_(c)degiusti.mid"
Melody, Rhythm = rhythm_of_melodies("../../dataset_git/Dataset/Bach/goldberg_variations_988_30_(c)degiusti.mid")



# def rhythm_of_melodies2(midi_path):
#     notes = []
#     rhythms = []
#
#     stream = converter.parse(midi_path)
#     codeword = stream.chordify()
#     striped_codeword = codeword.stripTies()
#
#     for c in striped_codeword.recurse().getElementsByClass('Chord'):
#         newChord = []
#         for n in c:
#             if (n.tie is None) or (n.tie.type == 'start'):
#                 newChord.append(n)
#         newChord = music21.chord.Chord(newChord)
#         if len(newChord.pitches) != 0:
#             notes.append(".".join(str(nc.pitch) for nc in newChord))
#             rhythms.append(str(newChord.quarterLength))
#     return notes, rhythms