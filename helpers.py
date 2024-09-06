import dill
import re
import os
import configure as conf


def load_data(element):
    dataInst_path = None
    if element == 'melody':
        dataInst_path = conf.dataInst_filename
    elif element == 'rhythm':
        dataInst_path = conf.dataInst_rhythm_filename
    with open(os.path.join(conf.dataInst_dir, dataInst_path), "rb") as f:
        data = dill.load(f)
    return data


def get_song_path_from_id(songID, dataInfo):
    composer = "".join(re.findall('[^0-9]', songID))
    song_name = list(dataInfo[dataInfo['songID']==songID]['song_name'])[0]
    return os.path.join(conf.dataset_dir, os.path.join(composer,song_name))


def progressBar(current, total, barLength=70):
    percent = float(current) * 100 / total
    arrow = '-' * int(percent/100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))
    print('[%s%s] %d %%' % (arrow, spaces, percent), end='\r')
