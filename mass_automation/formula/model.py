import torch
import pytorch_lightning as pl
from torch import nn
from torchmetrics import AUROC, MeanSquaredError, R2Score

from ..utils import Element, ELEMENT_DICT

LOW_MEMORY = True


class LinearWithHidden(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, activation=True, dropout=True):
        super().__init__()

        self.embedder = nn.Sequential(*([
                                            nn.Linear(in_size, hidden_size),
                                            nn.ReLU(),
                                            nn.Dropout(p=.5) if dropout else nn.BatchNorm1d(hidden_size),
                                            nn.Linear(hidden_size, out_size)] + ([nn.Sigmoid()] if activation else [])
                                        )
                                      )

    def forward(self, spectra):
        return self.embedder(spectra)


def log_elementwise(model, metric, prefix):
    for index, element in ELEMENT_DICT.items():
        model.log(f'{prefix}/{element}', metric[index - 1])


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_weights = torch.cuda.FloatTensor([
            1.00015549, 7.51934304, 3.23153067, 5.70698071, 2.2573169,
            1., 1.07670342, 1.04998035, 1.37827889, 9.,
            2.5069211, 3.75315276, 3.49343367, 1.96886256, 1.82790865,
            1.32515465, 1.47365777, 6.15073554, 3.09395335, 3.85379364,
            5.57853564, 3.53550584, 4.2966888, 3.96174946, 4.09297436,
            3.37820844, 3.70510843, 3.58163235, 3.35135713, 3.40528989,
            4.40474687, 3.73045542, 3.56244893, 3.06550287, 1.82807613,
            8.20392347, 4.42029638, 4.93821596, 4.24775148, 3.6310292,
            5.39355859, 4.31881343, 4.06921439, 3.60191452, 4.41534412,
            3.71343288, 4.16736028, 4.6824789, 4.41206426, 3.09379087,
            4.03909905, 3.97157701, 2.10576477, 6.86832305, 4.74140436,
            4.51107208, 5.13457492, 5.13457492, 5.60731637, 5.19140513,
            4.06921439, 5.41185936, 5.23881167, 4.96462708, 5.66331038,
            5.52426213, 6.001617, 5.67937615, 6.23822149, 5.25932782,
            5.79044175, 4.72382414, 5.43447359, 4.09589722, 5.16498326,
            4.86974745, 4.22316498, 3.5358764, 4.19837181, 4.02766207,
            4.67873498, 4.24337692, 4.36710707, 4.06921439, 4.06921439,
            4.06921439, 4.06921439, 4.06921439, 4.06921439, 5.69581094,
            7.8798335, 4.73730494, 4.06921439, 4.06921439, 4.06921439,
            4.06921439, 4.06921439, 4.06921439, 4.06921439, 4.06921439,
            4.06921439, 4.06921439, 4.06921439, 4.06921439, 4.06921439,
            4.06921439, 4.06921439, 4.06921439, 4.06921439, 4.06921439,
            4.06921439, 4.06921439, 4.06921439, 4.06921439, 4.06921439,
            4.06921439, 4.06921439, 4.06921439, 4.06921439
        ])

    def forward(self, y, y_hat):
        return (self.loss_weights * (y_hat - y) ** 2).mean()


class LSTM(pl.LightningModule):
    def __init__(self,

                 lstm_in_size=16,

                 lstm_hidden_size=128,
                 lstm_num_layers=1,
                 lstm_bidirectional=False,
                 lstm_dropout=0.5,

                 decoder_hidden_size=128,
                 activation=True,

                 loss='MSE',
                 opt='Adam',
                 lr=2e-4,

                 **kwargs
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            lstm_in_size,
            lstm_hidden_size,
            lstm_num_layers,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout
        )

        self.decoder = LinearWithHidden(
            lstm_hidden_size * (2 if lstm_bidirectional else 1) * lstm_num_layers,
            decoder_hidden_size,
            Element.n_elements,
            activation, True if 'dropout' not in kwargs else kwargs['dropout']
        )

        self.loss_name = loss

        if loss == 'MSE':
            self.loss = nn.MSELoss()
        elif loss == 'WeightedMSE':
            self.loss = WeightedMSELoss()
        elif loss == 'CE':
            self.loss = nn.BCELoss()
        else:
            raise ValueError(f'Unknown loss: {loss}')

        if loss.endswith('MSE'):
            self.train_r2 = R2Score(num_outputs=Element.n_elements, multioutput='raw_values')
            self.val_r2 = R2Score(num_outputs=Element.n_elements, multioutput='raw_values')
            self.test_r2 = R2Score(num_outputs=Element.n_elements, multioutput='raw_values')

        else:
            if not LOW_MEMORY:
                self.train_rocauc = AUROC(num_classes=Element.n_elements, average=None)
                self.val_rocauc = AUROC(num_classes=Element.n_elements, average=None)
                self.test_rocauc = AUROC(num_classes=Element.n_elements, average=None)

    def forward(self, x):
        _, (x, _) = self.lstm(x)
        x = self.decoder(x.permute(1, 0, 2).flatten(1))

        return x

    def training_step(self, x, batch_id):
        x, y = x

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss.item())

        if self.loss_name.endswith('MSE'):
            train_r2 = self.train_r2(y_hat, y)
            log_elementwise(self, train_r2, 'r2/train')
        else:
            if not LOW_MEMORY:
                train_rocauc = self.train_rocauc(y_hat, y.long())
                log_elementwise(self, train_rocauc, 'rocauc/train')

        return loss

    def train_epoch_end(self, outputs):
        if self.loss_name.endswith('MSE'):
            self.train_r2.reset()
        else:
            if not LOW_MEMORY:
                self.train_rocauc.reset()

    def validation_step(self, x, batch_id):
        x, y = x

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss.item())

        if self.loss_name.endswith('MSE'):
            val_r2 = self.val_r2(y_hat, y)
            log_elementwise(self, val_r2, 'r2/val')
        else:
            if not LOW_MEMORY:
                val_rocauc = self.val_rocauc(y_hat, y.long())
                log_elementwise(self, val_rocauc, 'rocauc/val')

        return loss

    def validation_epoch_end(self, outputs):
        if self.loss_name.endswith('MSE'):
            self.val_r2.reset()
        else:
            if not LOW_MEMORY:
                self.val_rocauc.reset()

    def test_step(self, x, batch_id):
        x, y = x

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss.item())

        if self.loss_name.endswith('MSE'):
            test_r2 = self.test_r2(y_hat, y)
            log_elementwise(self, test_r2, 'r2/test')
        else:
            if not LOW_MEMORY:
                test_rocauc = self.test_rocauc(y_hat, y.long())
                log_elementwise(self, test_rocauc, 'rocauc/test')
        return loss

    def test_epoch_end(self, outputs):
        if self.loss_name.endswith('MSE'):
            self.test_r2.reset()
        else:
            if not LOW_MEMORY:
                self.test_rocauc.reset()

    def configure_optimizers(self):
        if self.hparams.opt == 'Adam':
            return torch.optim.Adam(self.parameters(), self.hparams.lr)

        if self.hparams.opt == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), self.hparams.lr)

        raise ValueError


class MLP(pl.LightningModule):
    def __init__(self,

                 in_size=100,
                 hidden_size=50,

                 activation=True,

                 loss='MSE',
                 opt='Adam',
                 lr=2e-4,

                 **kwargs
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.mlp = LinearWithHidden(in_size, hidden_size, Element.n_elements, activation)

        self.loss_name = loss

        if loss == 'MSE':
            self.loss = nn.MSELoss()
        elif loss == 'WeightedMSE':
            self.loss = WeightedMSELoss()
        elif loss == 'CE':
            self.loss = nn.BCELoss()
        else:
            raise ValueError(f'Unknown loss: {loss}')

        if loss.endswith('MSE'):
            self.train_r2 = R2Score(num_outputs=Element.n_elements, multioutput='raw_values')
            self.val_r2 = R2Score(num_outputs=Element.n_elements, multioutput='raw_values')
            self.test_r2 = R2Score(num_outputs=Element.n_elements, multioutput='raw_values')

        else:
            if not LOW_MEMORY:
                self.train_rocauc = AUROC(num_classes=Element.n_elements, average=None)
                self.val_rocauc = AUROC(num_classes=Element.n_elements, average=None)
                self.test_rocauc = AUROC(num_classes=Element.n_elements, average=None)

    def forward(self, x):
        return self.mlp(x)

    def training_step(self, x, batch_id):
        x, y = x

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss.item())

        if self.loss_name.endswith('MSE'):
            train_r2 = self.train_r2(y_hat, y)
            log_elementwise(self, train_r2, 'r2/train')
        else:
            if not LOW_MEMORY:
                train_rocauc = self.train_rocauc(y_hat, y.long())
                log_elementwise(self, train_rocauc, 'rocauc/train')

        return loss

    def train_epoch_end(self, outputs):
        if self.loss_name.endswith('MSE'):
            self.train_r2.reset()
        else:
            if not LOW_MEMORY:
                self.train_rocauc.reset()

    def validation_step(self, x, batch_id):
        x, y = x

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss.item())

        if self.loss_name.endswith('MSE'):
            val_r2 = self.val_r2(y_hat, y)
            log_elementwise(self, val_r2, 'r2/val')
        else:
            if not LOW_MEMORY:
                val_rocauc = self.val_rocauc(y_hat, y.long())
                log_elementwise(self, val_rocauc, 'rocauc/val')

        return loss

    def validation_epoch_end(self, outputs):
        if self.loss_name.endswith('MSE'):
            self.val_r2.reset()
        else:
            if not LOW_MEMORY:
                self.val_rocauc.reset()

    def test_step(self, x, batch_id):
        x, y = x

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss.item())

        if self.loss_name.endswith('MSE'):
            test_r2 = self.test_r2(y_hat, y)
            log_elementwise(self, test_r2, 'r2/test')
        else:
            if not LOW_MEMORY:
                test_rocauc = self.test_rocauc(y_hat, y.long())
                log_elementwise(self, test_rocauc, 'rocauc/test')
        return loss

    def test_epoch_end(self, outputs):
        if self.loss_name.endswith('MSE'):
            self.test_r2.reset()
        else:
            if not LOW_MEMORY:
                self.test_rocauc.reset()

    def configure_optimizers(self):
        if self.hparams.opt == 'Adam':
            return torch.optim.Adam(self.parameters(), self.hparams.lr)

        if self.hparams.opt == 'RMSprop':
            return torch.optim.RMSprop(self.parameters(), self.hparams.lr)

        raise ValueError
