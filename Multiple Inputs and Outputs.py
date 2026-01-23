import torch # allows creation of tensors
import torch.nn as nn # allows us to make neural networks
import torch.nn.functional as F # for access to activation and loss functions
from torch.optim import Adam # optim has many Optimizers, Adam is one of them

import lightning as L # makes creating NN easier
from torch.utils.data import DataLoader, TensorDataset # for creating datasets and dataloaders

import pandas as pd # for normalization
from sklearn.model_selection import train_test_split # to split data into train and test sets

path = "Learning PyTorch/iris.txt"
df = pd.read_table(path, sep=",", header=None)

print(df.shape)

df.columns = ["sepal_length",
              "sepal_width",
              "petal_length",
              "petal_width",
              "class"]

print(df.head())
print(df['class'].nunique())
print(df['class'].unique())

for class_name in df['class'].unique():
    print(f"Class: {class_name}")
    print(class_name, ": ", sum(df['class'] == class_name), sep="") # equal split b/w all classes

print(df[["petal_width", "sepal_width"]].head())


input_values = df[["petal_width", "sepal_width"]]
label_values = df["class"]

#print(label_values.factorize()) 
# factorize returns a list of lists and we only need the first list so [0] will work

classes_as_numbers = label_values.factorize()[0] # convert class names to numbers
print(classes_as_numbers)


input_train, input_test, label_train , label_test = train_test_split(input_values,
                                                                     classes_as_numbers,
                                                                     test_size=0.25,
                                                                     stratify=classes_as_numbers,
                                                                     random_state=42)

# stratify ensures equal distribution of classes in train and test sets
# random_state ensures reproducibility of the split (for a specific num always the same split)

print(f"Input: {input_train.shape}, {input_test.shape}, \nLabel: {label_train.shape}, {label_test.shape}")

one_hot_label_train = F.one_hot(torch.tensor(label_train)).type(torch.float32) # one hot encode labels
print(one_hot_label_train[:5]) # printing some one hot encoded labels

max_val_input = input_train.max()
min_val_input = input_train.min()

# min-max normalization
input_train = (input_train - min_val_input) / (max_val_input - min_val_input)   
input_test = (input_test - min_val_input) / (max_val_input - min_val_input) 

input_train_tensors = torch.tensor(input_train.values).type(torch.float32)
input_test_tensors = torch.tensor(input_test.values).type(torch.float32)    

train_dataset = TensorDataset(input_train_tensors, one_hot_label_train)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)



class MultipleIO(L.LightningModule):

    def __init__(self):

        super().__init__()

        L.seed_everything(42)

        # Initializing weights and biases

        self.input_to_hidden = nn.Linear(in_features=2, out_features=2, bias=True) # we give 2 inputs which then go to 2 activation functions
        self.hidden_to_output = nn.Linear(in_features=2, out_features=3, bias=True) # 2 activations go to 3 outputs (3 classes)
        
        self.loss = nn.MSELoss(reduction='sum') # mean squared error loss

    
    def forward(self, input):
        hidden = self.input_to_hidden(input) # linear transformation
        output_values = self.hidden_to_output(torch.relu(hidden)) # activation function + linear transformation to output layer

        return output_values
    
    def configure_optimizers(self):
        # configuring optimizers consists of passing all weights and biases, which is done by self.parameters()

        return Adam(self.parameters(), lr=0.001) # Adam optimizer with learning rate of 0.01
    
    def training_step(self, batch, batch_idx):

        inputs, labels = batch # unpacking the batch
        outputs = self.forward(inputs) # forward pass
        loss = self.loss(outputs, labels) # calculating loss

        return loss

model = MultipleIO()

#printing name and value for each named parameter (weights and biases)
for name, param in model.named_parameters():
    print(name, torch.round(param.data, decimals=2))


trainer = L.Trainer(max_epochs=500, enable_progress_bar=True)
trainer.fit(model, train_dataloader)

predictions =  model(input_test_tensors)

predicted_labels = torch.argmax(predictions, dim=1) #dim = 1 means argmax for columns
print(predicted_labels)


# Now compare predicted_labels with test_labels to calculate accuracy
#       torch.eq() computes element-wise equality between two tensors.
#       label_test, however, is just an array, so we convert it to a tensor
#       before passing it in. torch.sum() then adds up all of the "True"
#       output values to get the number of correct predictions.
#       We then divide the number of correct predictions by the number of predicted values,
#       obtained with len(predicted_labels), to get the percentage of correct predictions

accuracy = torch.sum(torch.eq(predicted_labels, torch.tensor(label_test)))/ len(predicted_labels)
print(f"Accuracy: {accuracy*100:.2f}%")

path_to_checkpoint = trainer.checkpoint_callback.best_model_path
print(f"Best model saved at: {path_to_checkpoint}")

trainer = L.Trainer(max_epochs=1000, enable_progress_bar=True)
trainer.fit(model, train_dataloader, ckpt_path=path_to_checkpoint)

predictions =  model(input_test_tensors)
preducted_labels = torch.argmax(predictions, dim=1)
accuracy = torch.sum(torch.eq(predicted_labels, torch.tensor(label_test)))/ len(predicted_labels)
print(f"Accuracy after increasing to 1000 epochs: {accuracy*100:.2f}%")


# making prediction for new sample
new_sample = [0.2, 0.3] # petal_width = 0.2, sepal_width = 0.3
normalized_new_sample = (new_sample - min_val_input / (max_val_input - min_val_input))
normalized_new_sample_tensor = torch.tensor(normalized_new_sample).type(torch.float32)
prediction = model(normalized_new_sample_tensor)
print(prediction)

