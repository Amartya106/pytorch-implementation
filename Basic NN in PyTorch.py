import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class myNN(nn.Module):
    def __init__(self):

        super().__init__()

        self.w1 = torch.tensor(1.43)
        self.b1 = torch.tensor(-0.61)

        self.w2 = torch.tensor(2.63)
        self.b2 = torch.tensor(-0.27)

        self.w3 = torch.tensor(-3.89)
        self.w4 = torch.tensor(1.35)

    def forward(self, input_values):

        top_x_axis_values = (input_values * self.w1) + self.b1
        bottom_x_axis_values = (input_values * self.w2) + self.b2

        top_y_axis_values = F.relu(top_x_axis_values)
        bottom_y_axis_values = F.relu(bottom_x_axis_values)

        output_values = (top_y_axis_values * self.w3) + (bottom_y_axis_values * self.w4)

        return output_values
    
model = myNN()
doses = torch.tensor([0.0, 0.5, 1.0])
model(doses)
torch.round(model(doses), decimals=2)

input_doses = torch.linspace(start = 0, end = 1, steps = 11)

top_x_axis_values = (input_doses*model.w1) + model.b1
print(top_x_axis_values)

top_y_axis_values = F.relu(top_x_axis_values)
print(top_y_axis_values)

final_top_y_axis_values = top_y_axis_values*model.w3


sns.set_style("whitegrid")

sns.scatterplot(x=input_doses.numpy(),
                y=final_top_y_axis_values.numpy(),
                color='blue',
                s=200)

sns.lineplot(x=input_doses.numpy(),
             y=final_top_y_axis_values.numpy(),
             color='blue',
             linewidth=2.5)


plt.ylabel("Top ReLU Output")
plt.xlabel("Input Dose")
plt.savefig("top_relu_output.png")


bottom_x_axis_values = (input_doses*model.w2) + model.b2
print(bottom_x_axis_values)

bottom_y_axis_values = F.relu(bottom_x_axis_values)
print(bottom_y_axis_values)

final_bottom_y_axis_values = bottom_y_axis_values*model.w4

sns.set_style("whitegrid")

sns.scatterplot(x=input_doses.numpy(),
                y=final_bottom_y_axis_values.numpy(),
                color='orange',
                s=200)

sns.lineplot(x=input_doses.numpy(),
             y=final_bottom_y_axis_values.numpy(),
             color='orange',
             linewidth=2.5)

plt.xlabel("Input Dose")
plt.ylabel("Bottom ReLU Output")
plt.savefig("bottom_relu_output.png")

plt.clf()

final_y_axis_values = final_top_y_axis_values+final_bottom_y_axis_values

sns.set_style("whitegrid")

sns.scatterplot(x = input_doses.numpy(),
                y = final_y_axis_values.numpy(),
                color='green',  
                s=200)

sns.lineplot(x = input_doses.numpy(),
                y = final_y_axis_values.numpy(),
                color='green',  
                linewidth=2.5)

plt.xlabel("Input Dose")
plt.ylabel("Final Output")
plt.savefig("final_output.png")






