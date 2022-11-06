from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import torch
import numpy as np
import glob
import os
import augment


def get_all_file_path(input_dir, file_extension):
    temp = glob.glob(os.path.join(input_dir, '**', '*.{}'.format(file_extension)), recursive=True)
    return temp

def get_random_start_point(size, audio_window):
    return np.random.randint(size - audio_window + 1)

def read_txt2list(file_path):
    with open(file_path, 'r') as data:
        file_list = [x.strip() for x in data.readlines()]
    return file_list

def audio_adjust_length(x, audio_window, fit=False):
    length_adj = audio_window - len(x[0])
    if length_adj > 0:
        half_adj = length_adj // 2
        x = F.pad(x, (half_adj, length_adj - half_adj))
    audio_length = len(x[0])
    if fit:
        random_index = np.random.randint(audio_length - audio_window + 1)
        x = x[:, random_index: random_index + audio_window]
    return x

def random_cutoff(waveform, audio_window, index=None):
    audio_length = waveform.shape[1]
    if index is None:
        random_index = np.random.randint(audio_length - audio_window + 1)
    else:
        random_index = index
    cutoff_waveform = waveform[:, random_index: random_index + audio_window]
    return cutoff_waveform

def random_clip_chain(waveform, sr):
    random_clip = np.random.randint(2, 11) * 0.1
    clip_chain = augment.EffectChain().clip(random_clip)
    return clip_chain.apply(waveform, src_info={'rate': sr})

def random_reverb_chain(waveform, sr):
    random_room_size = np.random.randint(0, 101)
    reverb_chain = augment.EffectChain().reverb(50, 50, random_room_size).channels(1)
    return reverb_chain.apply(waveform, src_info={'rate': sr})

def random_pitch_shift_chain(waveform, sr):
    random_pitch_shift = np.random.randint(-400, 400)
    pitch_shift_chain = augment.EffectChain().pitch(random_pitch_shift)
    return pitch_shift_chain.apply(waveform, src_info={'rate': sr})

def random_time_drop_chain(waveform, sr):
    random_time = np.random.randint(1, 5) * 0.1
    time_chain = augment.EffectChain().time_dropout(max_seconds=random_time)
    return time_chain.apply(waveform, src_info={'rate': sr})

def random_additive_background(waveform, sr):
    datalist_path = "./dataset/musan-total.txt"
    def noise_generator():
        dataset_path = datalist_path
        filelist = read_txt2list(dataset_path)
        pick = np.random.randint(len(filelist))
        source, sampling_rate = torchaudio.load(filelist[pick][4:])
        source = audio_adjust_length(source, len(waveform[0]))
        source = random_cutoff(source, len(waveform[0]))
        return source[0]
    random_snr = np.random.randint(10)+5
    background_chain = augment.EffectChain().additive_noise(noise_generator, snr=random_snr)
    return background_chain.apply(waveform, src_info={'rate': sr})

def random_additive_noise(waveform, sr):
    noise_generator = lambda: torch.zeros_like(waveform).uniform_()
    random_snr = np.random.randint(10) + 5
    noise_chain = augment.EffectChain().additive_noise(noise_generator, snr=random_snr)
    return noise_chain.apply(waveform, src_info={'rate': sr})


class UnlabeledWaveform(Dataset):
    def __init__(self, file_path, segment_size=20480, sampling_rate=16000):
        super(UnlabeledWaveform, self).__init__()
        self.file_list = []
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.file_list = read_txt2list(file_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index][4:]
        waveform01, sr = torchaudio.load(file_name)
        waveform02, sr = torchaudio.load(file_name)
        assert (
                self.sampling_rate == sr
        ), "sampling rate is not consistent throughout the dataset"

        waveform01 = audio_adjust_length(waveform01, self.segment_size, False)
        waveform02 = audio_adjust_length(waveform02, self.segment_size, False)
        pick = get_random_start_point(waveform01.shape[1], self.segment_size)
        waveform01 = waveform01[:, pick: pick + self.segment_size]
        waveform02 = waveform02[:, pick: pick + self.segment_size]

        waveform01 = random_clip_chain(waveform01, self.sampling_rate)
        waveform02 = random_clip_chain(waveform02, self.sampling_rate)

        waveform01 = random_reverb_chain(waveform01, self.sampling_rate)
        waveform02 = random_reverb_chain(waveform02, self.sampling_rate)

        waveform01 = random_pitch_shift_chain(waveform01, self.sampling_rate)
        waveform02 = random_pitch_shift_chain(waveform02, self.sampling_rate)

        waveform01 = audio_adjust_length(waveform01, self.segment_size, True)
        waveform02 = audio_adjust_length(waveform02, self.segment_size, True)

        waveform01 = random_time_drop_chain(waveform01, self.sampling_rate)
        waveform02 = random_time_drop_chain(waveform02, self.sampling_rate)

        waveform01 = random_additive_background(waveform01, self.sampling_rate)
        waveform02 = random_additive_background(waveform02, self.sampling_rate)

        return (waveform01, waveform02), 0


def main():
    dataset = UnlabeledWaveform(file_path='../dataset/FSD50K.dev_audio_16k.txt', segment_size=20480, sampling_rate=16000)
    print(len(dataset))
    for data in dataset:
        pass

if __name__ == '__main__':
    main()










