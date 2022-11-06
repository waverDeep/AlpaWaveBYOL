import copy
from adamp import AdamP
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def chunk_and_cat_step(x):
    chunks = x.chunk(3, dim=1)
    out_cat = torch.stack(chunks, dim=1)
    return out_cat

def set_requires_grad(model, requires):
    for parameter in model.parameters():
        parameter.requires_grad = requires

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def loss_function(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class FeatureExtractionStep(nn.Module):
    def __init__(self, input_dim, hidden_dim, stride, filter_size, padding):
        super(FeatureExtractionStep, self).__init__()
        assert (
                len(stride) == len(filter_size) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.feature_extraction_step = nn.Sequential()
        for index, (stride, filter_size, padding) in enumerate(zip(stride, filter_size, padding)):
            self.feature_extraction_step.add_module(
                "extraction_layer_{}".format(index),
                nn.Sequential(
                    nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                              kernel_size=filter_size, stride=stride, padding=padding),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                )
            )
            input_dim = hidden_dim

    def forward(self, x):
        return self.feature_extraction_step(x)


class FeatureEncodingStep(nn.Module):
    def __init__(self, input_dim, hidden_dim, stride, filter_size, padding):
        super(FeatureEncodingStep, self).__init__()
        assert (
                len(stride) == len(filter_size) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"

        self.feature_encoding_step = nn.Sequential()
        for index, (hidden_dim, stride, filter_size, padding) in enumerate(zip(hidden_dim, stride, filter_size, padding)):
            self.feature_encoding_step.add_module(
                "encoder_layer_{}".format(index),
                nn.Sequential(
                    nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim,
                              kernel_size=filter_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(),
                )
            )
            input_dim = hidden_dim

    def forward(self, x):
        return self.feature_encoding_step(x)


class Encoder(nn.Module):
    def __init__(self, feature_extraction_step, feature_encoding_step, concat_step):
        super(Encoder, self).__init__()
        self.feature_extraction_step = feature_extraction_step
        self.feature_encoding_step = feature_encoding_step
        self.adaptive_average_pool = nn.AdaptiveAvgPool2d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.concat_step = concat_step

    def forward(self, x):
        out = F.normalize(x, dim=-1, p=2)
        out = self.feature_extraction_step(out)

        out = chunk_and_cat_step(out)

        out = F.normalize(out, dim=-1, p=2)
        out = self.feature_encoding_step(out)

        out = F.normalize(out, dim=-1, p=2)
        out_avg = self.adaptive_average_pool(out)
        out_max = self.adaptive_max_pool(out)

        out_avg = self.flatten(out_avg)
        out_max = self.flatten(out_max)

        out = out_avg + out_max
        out = self.concat_step(out)
        return out


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPLayer, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.MLP(x)


class Projector(nn.Module):
    def __init__(self, mlp_layer):
        super(Projector, self).__init__()
        self.mlp_layer = mlp_layer

    def forward(self, x):
        return self.mlp_layer(x)

class Predictor(nn.Module):
    def __init__(self, mlp_layer):
        super(Predictor, self).__init__()
        self.mlp_layer = mlp_layer

    def forward(self, x):
        return self.mlp_layer(x)


class LinearClassifier(nn.Module):
    def __init__(self, mlp_layer):
        super(LinearClassifier, self).__init__()
        self.mlp_layer = mlp_layer

    def forward(self, x):
        return self.mlp_layer(x)


class WaveBYOL(pl.LightningModule):
    def __init__(self, encoder, projector, predictor, ema_updater):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor
        
        self.target_encoder = None
        self.target_projector = None
        self.ema_updater = ema_updater

    def forward(self, x):
        out = self.encoder(x)
        return out

    def configure_optimizers(self):
        adamp = AdamP(self.parameters(),
             lr=0.0001,
             betas=[0.9, 0.999],
             weight_decay=1.5e-6,
             eps=1e-3,
        )
        return adamp

    def training_step(self, batch, batch_idx):
        samples, y = batch
        sample01 = samples[0]
        sample02 = samples[1]

        online01_out = self.encoder(sample01)
        online01_out = self.projector(online01_out)
        online01_out = self.predictor(online01_out)
        online02_out = self.encoder(sample02)
        online02_out = self.projector(online02_out)
        online02_out = self.predictor(online02_out)
        
        with torch.no_grad():
            target01_out = self.target_encoder(sample01)
            target01_out = self.target_projector(target01_out)
            target02_out = self.target_encoder(sample02)
            target02_out = self.target_projector(target02_out)

        loss01 = loss_function(online01_out, target02_out)
        loss02 = loss_function(online02_out, target01_out)
        loss = loss01 + loss02
        loss = loss.mean()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        samples, y = batch
        sample01 = samples[0]
        sample02 = samples[1]

        with torch.no_grad():
            online01_out = self.encoder(sample01)
            online01_out = self.projector(online01_out)
            online01_out = self.predictor(online01_out)
            online02_out = self.encoder(sample02)
            online02_out = self.projector(online02_out)
            online02_out = self.predictor(online02_out)

            target01_out = self.target_encoder(sample01)
            target01_out = self.target_projector(target01_out)
            target02_out = self.target_encoder(sample02)
            target02_out = self.target_projector(target02_out)

            loss01 = self.loss_function(online01_out, target02_out)
            loss02 = self.loss_function(online02_out, target01_out)
            loss = loss01 + loss02
            loss = loss.mean()

            return loss



    def on_epoch_start(self):
        self.target_encoder = copy.deepcopy(self.encoder)
        set_requires_grad(self.target_encoder, False)
        self.target_projector = copy.deepcopy(self.projector)
        set_requires_grad(self.target_projector, False)

    def on_epoch_end(self):
        update_moving_average(self.ema_updater, self.target_encoder, self.encoder)
        update_moving_average(self.ema_updater, self.target_projector, self.projector)


def main():
    input_sample = torch.randn(2, 1, 16000)
    feature_extraction_step = FeatureExtractionStep(
        input_dim=1,
        hidden_dim=513,
        stride=[5, 4, 2, 2, 2],
        filter_size=[10, 8, 4, 2, 2],
        padding=[2, 2, 2, 2, 1]
    )
    feature_encoding_step = FeatureEncodingStep(
        input_dim=3,
        hidden_dim=[64, 128, 256, 512, 1024],
        stride=[5, 3, 2, 2, 1],
        filter_size=[10, 8, 4, 4, 3],
        padding=[1, 1, 1, 1, 1]
    )
    concat_step = nn.Linear(1024, 2048)
    encoder = Encoder(feature_extraction_step, feature_encoding_step, concat_step)
    print(encoder)
    out_sample = encoder(input_sample)
    print(out_sample.size())

    projector = Projector(MLPLayer(2048, 2048, 2048))
    predictor = Predictor(MLPLayer(2048, 2048, 2048))
    ema_updater = EMA(0.99)

    model = WaveBYOL(encoder=encoder, projector=projector, predictor=predictor, ema_updater=ema_updater)
    print(model)
    out_sample = model(input_sample)
    print(out_sample.size())


if __name__ == '__main__':
    main()










