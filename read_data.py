import os
import pandas as pd
from data_generator import Preprocessing
from data_generator_update import Preprocessing_update
import dill
import argparse
import configure as conf
from helpers import progressBar


def main(args):
    dataInfo = ReadDataInfo(args.data_info_dir)
    dataInst = ReadDataInstance(dataInfo, args.data_dir, args.element)
    print("========================== Saving dataset ==========================")
    save_filename = ''
    if args.element == 'melody':
        save_filename = conf.dataInst_filename
    elif args.element == 'rhythm':
        save_filename = conf.dataInst_rhythm_filename
    with open(os.path.join(args.save_dir, save_filename), "wb") as f:
        dill.dump(dataInst, f)

def ReadDataInfo(dataInfo_dir):
    dataInfo = pd.read_excel(os.path.join(dataInfo_dir, conf.dataInfo_filename), index_col=0)
    return dataInfo

def ReadDataInstance(dataInfo, data_dir, encoding_element):
    print("========================== Preprocessing dataset ==========================")
    dataInst = {}
    for i in range(dataInfo.shape[0]):
        id = dataInfo['songID'][i]
        comp = dataInfo['composer'][i]
        song_name = dataInfo['song_name'][i]
        try:
            if encoding_element == 'melody':
                dataInst[id] = Preprocessing(os.path.join(os.path.join(data_dir, comp), song_name))
            elif encoding_element == 'rhythm':
                dataInst[id] = Preprocessing_update(os.path.join(os.path.join(data_dir, comp), song_name))
            print(i+1, ' / ', dataInfo.shape[0])
        except Exception as ex:
            print('--------fail-to-read-the-data-', id, '--------')
            print(ex)
        progressBar(i, dataInfo.shape[0])
    return dataInst



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument('--element', type=str, required=True, help='which element to encode data, rhythm or melody')
    parser.add_argument('--data-dir', type=str, default=conf.dataset_dir, help='directory of MIDI files to analyze')
    parser.add_argument('--data-info_dir', type=str, default=conf.dataInfo_dir, help='directory of excel file having information of the dataset')
    parser.add_argument('--save-dir', type=str, default=conf.dataInst_dir, help='saving directory of processed dataset')
    args = parser.parse_args()

    main(args)

