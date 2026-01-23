import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import lightning as L
from torch.utils.data import DataLoader, TensorDataset

training_inputs = torch.tensor([[0.0], [0.5], [1.0]])
training_labels = torch.tensor([[0.0], [1.0], [0.0]])

training_dataset = TensorDataset(training_inputs, training_labels)
dataloader = DataLoader(training_dataset) # by default batch size is 1

class myNN(L.LightningModule):
    def __init__(self):

        super().__init__()

        self.w1 = nn.Parameter(torch.tensor(0.06))
        self.b1 = nn.Parameter(torch.tensor(0.0))

        self.w2 = nn.Parameter(torch.tensor(3.49))
        self.b2 = nn.Parameter(torch.tensor(0.0)) 

        self.w3 = nn.Parameter(torch.tensor(-4.11))
        self.w4 = nn.Parameter(torch.tensor(2.74))

        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, input_values):

        top_x_axis_values = (input_values * self.w1) + self.b1
        bottom_x_axis_values = (input_values * self.w2) + self.b2

        top_y_axis_values = F.relu(top_x_axis_values)
        bottom_y_axis_values = F.relu(bottom_x_axis_values)

        output_values = (top_y_axis_values * self.w3) + (bottom_y_axis_values * self.w4)

        return output_values
    
    def configure_optimizers(self):
        return SGD(self.parameters(), lr=0.01)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        ouputs = self.forward(inputs)
        loss = self.loss(ouputs, labels)

        return loss

model = myNN()

for name, param in model.named_parameters():
    print(name, torch.round(param.data, decimals=2))


output_values = model(training_inputs)
print(torch.round(output_values, decimals=2))

input_doses = torch.linspace(start = 0, end = 1, steps = 11)
output_values = model(input_doses)
print(output_values)

sns.set_style("whitegrid")

sns.scatterplot(x = input_doses.numpy(),
                y = output_values.detach().numpy(),
                color = 'blue',
                s = 200)

sns.lineplot(x = input_doses.numpy(),
             y = output_values.detach().numpy(),
             color = 'blue',
             linewidth = 2.5)

sns.scatterplot(x=training_inputs.numpy().flatten(),
                y=training_labels.numpy().flatten(),
                color='red',
                s=100)


plt.xlabel("Input Dose")
plt.ylabel("Model Output")
plt.savefig("model_output_before_training.png")

model = myNN()

trainer = L.Trainer(max_epochs = 500,
                    enable_progress_bar = True,
                    enable_checkpointing = True,
                    enable_model_summary = True)

trainer.fit(model, dataloader)

output_values = model(input_doses)
torch.round(output_values, decimals=2)

sns.set_style("whitegrid")

sns.scatterplot(x = input_doses.numpy(),
                y = output_values.detach().numpy(),
                color = 'blue',
                s = 200)    

sns.lineplot(x = input_doses.numpy(),
             y = output_values.detach().numpy(),
             color = 'blue',
             linewidth = 2.5)

sns.scatterplot(x=training_inputs.numpy().flatten(),
                y=training_labels.numpy().flatten(),
                color='red',
                s=100)

plt.xlabel("Input Dose")
plt.ylabel("Model Output")
plt.savefig("model_output_after_training.png")

