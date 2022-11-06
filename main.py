import models.model as model
import data.dataset as dataset
import torch.nn as nn
import pytorch_lightning as pl
import torch.utils.data as data


def main():
    feature_extraction_step = model.FeatureExtractionStep(
        input_dim=1,
        hidden_dim=513,
        stride=[5, 4, 2, 2, 2],
        filter_size=[10, 8, 4, 2, 2],
        padding=[2, 2, 2, 2, 1]
    )
    feature_encoding_step = model.FeatureEncodingStep(
        input_dim=3,
        hidden_dim=[64, 128, 256, 512, 1024],
        stride=[5, 3, 2, 2, 1],
        filter_size=[10, 8, 4, 4, 3],
        padding=[1, 1, 1, 1, 1]
    )
    concat_step = nn.Linear(1024, 2048)
    encoder = model.Encoder(feature_extraction_step, feature_encoding_step, concat_step)

    projector = model.Projector(model.MLPLayer(2048, 2048, 2048))
    predictor = model.Predictor(model.MLPLayer(2048, 2048, 2048))
    ema_updater = model.EMA(0.99)

    wave_byol = model.WaveBYOL(encoder=encoder, projector=projector, predictor=predictor, ema_updater=ema_updater)
    train_dataset = dataset.UnlabeledWaveform(file_path='./dataset/FSD50K.dev_audio_16k.txt', segment_size=20480,
                                sampling_rate=16000)
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
    )
    trainer = pl.Trainer(gpus=2, precision=16)
    trainer.fit(wave_byol, train_dataloader)

if __name__ == '__main__':
    main()

