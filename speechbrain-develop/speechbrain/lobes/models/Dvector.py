import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm1d(nn.Module):
    def __init__(
            self,
            input_shape=None,
            input_size=None,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            combine_batch_time=False,
            skip_transpose=False
    ):
        super().__init__()

        self.combine_batch_time = combine_batch_time
        self.skip_transpose = skip_transpose

        if input_size is None and skip_transpose:
            input_size = input_shape[1]
        elif input_size is None:
            input_size = input_shape[-1]

        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x):
        shape_or = x.shape
        if self.combine_batch_time:
            if x.ndim == 3:
                x = x.reshape(shape_or[0] * shape_or[1], shape_or[2])
            else:
                x = x.reshape(
                    shape_or[0] * shape_or[1], shape_or[3], shape_or[2]
                )

        elif not self.skip_transpose:
            x = x.transpose(-1, 1)

        x_n = self.norm(x)

        if self.combine_batch_time:
            x_n = x_n.reshape(shape_or)
        elif not self.skip_transpose:
            x_n = x_n.transpose(1, -1)

        return x_n


class DNN_Block(nn.Module):
    def __init__(self, input_size, lin_neurons):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(
            nn.Linear(input_size, lin_neurons)
        )
        self.blocks.append(
            BatchNorm1d(input_size=lin_neurons)
        )
        self.blocks.append(nn.ReLU())

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x


class Dvector(nn.Module):
    def __init__(self, input_size=80, hidden_size=512, lin_neurons=512):
        super().__init__()

        self.in_linear = DNN_Block(input_size, hidden_size)

        self.hidden_linear1 = DNN_Block(hidden_size, hidden_size)

        self.hidden_linear2 = DNN_Block(hidden_size, hidden_size)

        self.hidden_linear3 = DNN_Block(hidden_size, hidden_size)

        self.dropout1 = nn.Dropout(p=0.15)

        self.hidden_linear4 = DNN_Block(hidden_size, hidden_size)

        self.dropout2 = nn.Dropout(p=0.15)

        self.fc = nn.Linear(hidden_size, lin_neurons)

    def forward(self, x, lengths=None):
        x = self.in_linear(x)
        x = self.hidden_linear1(x)
        x = self.hidden_linear2(x)
        x = self.hidden_linear3(x)
        x = self.dropout1(x)
        x = self.hidden_linear4(x)
        x = self.dropout2(x)

        # embedding
        x = self.fc(x)
        x = x.mean(dim=1)

        return x.unsqueeze(1)



# class dvector(nn.Module):
#     def __init__(self, in_dim=80, hidden_dim=256, k=2, num_classes=512):
#         super(dvector, self).__init__()
#         self.in_linear = nn.ModuleList([
#             nn.Linear(in_dim, hidden_dim)
#             for _ in range(k)
#         ])

#         self.hid_linear1 = nn.ModuleList([
#             nn.Linear(hidden_dim, hidden_dim)
#             for _ in range(k)
#         ])

#         self.hid_linear2 = nn.ModuleList([
#             nn.Linear(hidden_dim, hidden_dim)
#             for _ in range(k)
#         ])

#         self.hid_linear3 = nn.ModuleList([
#             nn.Linear(hidden_dim, hidden_dim)
#             for _ in range(k)
#         ])
#         self.dropout1 = nn.Dropout(p=0.5)

#         self.hid_linear4 = nn.ModuleList([
#             nn.Linear(hidden_dim, hidden_dim)
#             for _ in range(k)
#         ])
#         self.dropout2 = nn.Dropout(p=0.5)

#         self.out_linear = nn.Linear(hidden_dim, num_classes)

#     def forward(self, x):
#         #x: (B, T, D)

#         x = torch.stack([sub(x) for sub in self.in_linear], dim=1)
#         x = torch.max(x, dim=1)[0]

#         x = torch.stack([sub(x) for sub in self.hid_linear1], dim=1)
#         x = torch.max(x, dim=1)[0]

#         x = torch.stack([sub(x) for sub in self.hid_linear3], dim=1)
#         x = torch.max(x, dim=1)[0]

#         x = torch.stack([sub(x) for sub in self.hid_linear3], dim=1)
#         x = torch.max(x, dim=1)[0]
#         x = self.dropout1(x)

#         x = torch.stack([sub(x) for sub in self.hid_linear4], dim=1)
#         x = torch.max(x, dim=1)[0]
#         x = self.dropout2(x)

#         x = x.mean(dim=1)
#         embedding = x
        
#         return embedding