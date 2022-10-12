import numpy as np
import torch
from torch import nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
import torch_xla.utils.utils as xu
from torch_xla.experimental.xla_sharding import Mesh
import torch.optim as optim
from torch import nn

# Model params = 524,288
# Model param size = 8bytes (assumed)
# Loading the model requires 4,194,304 ~ 4MiB
# Input size is 8KB per entry.
# Need 4 000 000 examples in RAM to exhaust memory
INPUT_DIM = 1024
HIDDEN_DIM = INPUT_DIM // 2

BATCH_SIZE = 128


class SimpleLinear(nn.Module):

  def __init__(self):
    super(SimpleLinear, self).__init__()
    self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(HIDDEN_DIM, 1)

  def forward(self, x):
    y = self.relu(self.fc1(x))
    z = self.fc2(y)
    return z


device = xm.xla_device()

# define mesh:Mesh
num_devices = len(xm.get_xla_supported_devices())
print('num_devices: ', num_devices)
mesh_shape = (2, int(num_devices / 2))
print('mesh_shape: ', mesh_shape)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('x', 'y'))
print('logical_mesh:\n', mesh.get_logical_mesh())


def train():
  print('===> Preparing data..')
  train_dataset_len = 128 * 128
  num_epochs = 18
  lr = 0.1
  train_loader = xu.SampleGenerator(
      data=(torch.zeros(BATCH_SIZE,
                        INPUT_DIM), torch.zeros(BATCH_SIZE, dtype=torch.int64)),
      sample_count=train_dataset_len // BATCH_SIZE)
  torch.manual_seed(42)
  model = SimpleLinear().to(device)
  xs.mark_sharding(model.fc1.weight, mesh, (0, 1))

  optimizer = optim.SGD(model.parameters(), lr=lr)

  loss_fn = nn.CrossEntropyLoss()

  def train_loop_fn(loader, epoch):
    model.train()
    for step, (data, target) in enumerate(loader):
      data = data.to(device)
      target = target.to(device)
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      optimizer.step()
      xm.mark_step()
      if step % 10 == 0:
        print(f"Epoch {epoch} step {step} loss {loss}")

  for epoch in range(20):
    train_loop_fn(train_loader, epoch)

  return model


print('Start training loop...')
# xm.set_replication(xm.xla_device(), xm.get_xla_supported_devices())
m = train()
t = torch.randn(HIDDEN_DIM, INPUT_DIM).to(device)
m(t).cpu()
