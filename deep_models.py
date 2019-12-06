import torch
from torch.nn import Dropout2d, MaxPool2d, Conv2d, ConvTranspose2d
from torch.nn.functional import relu
from torch.nn import ModuleList, MSELoss
from torch.optim import Adam
from tqdm import tqdm


def transform_into_tensor(image):
    return torch.tensor(image).unsqueeze(dim=-3)


def train_model(model, x, max_delay, lr, batch_size, cuda=False):
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fun = MSELoss()
    model.train()
    if torch.cuda.is_available() and cuda:
        model.cuda()

    for epoch in range(1, max_delay+1):
        y_train = x[epoch:]
        x_train = x[:-epoch]
        losses = []
        for i in tqdm(range(0, len(x_train), batch_size)):
            x_batch = x_train[i: min(i + batch_size, len(x_train))]
            y_batch = y_train[i: min(i + batch_size, len(x_train))]
            if torch.cuda.is_available() and cuda:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            pred = model(x_batch)
            loss = loss_fun(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
        losses = torch.cat(losses)
        print(f"epoch: {epoch}, average_loss: {losses.mean()}")
    return model.cpu()


class ConvLayer(torch.nn.Module):
    def __init__(self, n_in_channel, n_out_channel, kernel, dropout=0):
        super().__init__()
        self.drop1 = Dropout2d(p=dropout)
        self.conv1 = Conv2d(n_in_channel, n_out_channel, kernel, padding=max(kernel-2, 0))
        self.drop2 = Dropout2d(p=dropout)
        self.conv2 = Conv2d(n_out_channel, n_out_channel, kernel, padding=max(kernel-2, 0))

    def forward(self, x):
        x = self.conv1(self.drop1(x))
        x = relu(x)
        x = self.conv2(self.drop2(x))
        return relu(x)


class DownLayer(torch.nn.Module):
    def __init__(self, n_in_channel, n_out_channel, conv_kernel=3, dropout=0, pool_kernel=2):
        super().__init__()
        self.max_pool = MaxPool2d(pool_kernel)
        self.conv = ConvLayer(n_in_channel, n_out_channel, conv_kernel, dropout)

    def forward(self, x):
        x = self.conv(x)
        x_pool = self.max_pool(x)
        return x_pool, x


class UpLayer(torch.nn.Module):
    def __init__(self, n_in_channel, n_out_channel, conv_kernel=3, up_kernel=2, up_stride=2, dropout=0):
        super().__init__()
        self.dropout = Dropout2d(p=dropout)
        self.up_conv = ConvTranspose2d(n_in_channel, n_out_channel, up_kernel, stride=up_stride)
        self.conv = ConvLayer(2*n_out_channel, n_out_channel, conv_kernel, dropout)

    def forward(self, x, x_old_crop):
        x = self.dropout(x)
        x = self.up_conv(x)
        x = relu(x)
        x = torch.cat([x_old_crop, x], dim=-3)
        return self.conv(x)


class UNet(torch.nn.Module):
    def __init__(self, image_channels, input_channels, output_channels, vertical_levels=3, dropout=0):
        super().__init__()
        self.embedding_layer = Conv2d(image_channels, input_channels, kernel_size=3, padding=1)
        encoder_channels = [input_channels*(2**i) for i in range(vertical_levels)]
        self.encoder_layers = ModuleList([DownLayer(encoder_channels[i], encoder_channels[i+1], dropout=dropout)
                                          for i in range(vertical_levels-1)])

        self.middle_layer = ConvLayer(encoder_channels[-1], 2*encoder_channels[-1], kernel=3, dropout=dropout)

        decoder_channels = [int(2*encoder_channels[-1]/(2**i)) for i in range(vertical_levels)]
        self.decoder_layers = ModuleList([UpLayer(decoder_channels[i], decoder_channels[i+1], dropout=dropout)
                                          for i in range(vertical_levels-1)])
        self.output_layer = Conv2d(decoder_channels[-1], output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = relu(self.embedding_layer(x))
        xs_list = []
        for layer in self.encoder_layers:
            x, x_to_store = layer(x)
            xs_list.append(x_to_store)

        x = relu(self.middle_layer(x))
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, xs_list[-(i+1)])

        return self.output_layer(x)


if __name__ == '__main__':
    from input_reader import read_simulation_file, get_all_time_ids
    import yaml
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)['data_path']
    simulation_path = cfg['simulation_path']
    save_path = cfg['save_path']
    field = 'temperature'
    model = UNet(1, 32, 1)
    time_ids = get_all_time_ids(simulation_path)
    images = read_simulation_file(simulation_path, field, time_ids)
    tensor_images = transform_into_tensor(images)
    model = train_model(model, tensor_images, 5, 1e-4, 5)
    torch.save(model, save_path+'model.pt')
