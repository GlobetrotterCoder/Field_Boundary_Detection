from imports import *
from data import data
from dataloader import *
from utils import accuracy



IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

num_epochs  = 250

#Initializing wandb 

wandb.login()
wandb.init(project="Ai4boundaries_ResUnet")


import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,                      # model output channels (number of classes in your dataset)
)


# Specify the input shape of your model (assuming input size of (3, 224, 224))
input_shape = (3, 256, 256)

# Move the model to a device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Get the summary of the model
print(summary(model, input_shape))

X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_val = np.load('X_val.npy')
Y_val = np.load('Y_val.npy')
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

Y_train = np.expand_dims(Y_train, axis=-1)
Y_val = np.expand_dims(Y_val, axis=-1)
Y_test = np.expand_dims(Y_test, axis=-1)

# Get the shape of X_train
X_train_shape = X_train.shape
Y_train_shape = Y_train.shape
X_val_shape = X_val.shape
Y_val_shape = Y_val.shape
X_test_shape = X_test.shape
Y_test_shape = Y_test.shape


print("Shape of X_train",X_train_shape)
print("Shape of Y_train",Y_train_shape)
print("Shape of X_val",X_val_shape)
print("Shape of Y_val",Y_val_shape)
print("Shape of X_test",X_test_shape)
print("Shape of Y_test",Y_test_shape)

#Getting the three dataloaders

# Create custom datasets
train_dataset = CustomDataset(X_train, Y_train)
val_dataset = CustomDataset(X_val, Y_val)
test_dataset = CustomDataset(X_test,Y_test)

batch_size = 64
train_loader,val_loader,test_loader = dataloader(train_dataset,val_dataset,test_dataset,batch_size)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCEWithLogitsLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


total_training_time = 0.0
global best_loss
best_loss = 1000
best_model_path = 'resunet_model_ai4boundaries.pt'


for epoch in range(num_epochs):
    model.train()
    running_loss = []
    train_accuracy = []
    
    for i,batch in enumerate(train_loader):
            
            inputs, targets = batch[0].to(device), batch[1].to(device)
            targets = torch.permute(targets,(0,3,1,2)).float()
            optimizer.zero_grad()
            inputs= torch.permute(inputs,(0,3,1,2))
            inputs = inputs.float()
            outputs = model(inputs)
            loss = loss_fn(outputs,targets)
            running_loss.append(loss.item())
            train_accuracy.append(accuracy(outputs,targets))
            loss.backward()
            optimizer.step()



    #  Calculate average training loss and accuracy
    average_train_loss = np.mean(np.array(running_loss))
    average_train_accuracy = np.mean(np.array(train_accuracy))

    print('average_train_loss:',average_train_loss)
    print('average train_accuracy:',average_train_accuracy)


    # Log training loss and accuracy to Wandb
    wandb.log({"Training Loss": average_train_loss, "Training Accuracy": average_train_accuracy}, step=epoch)
   
    # Evaluate on validation set after each epoch
    model.eval()
    val_loss = []
    val_accuracy = []

    
    with torch.no_grad():
        total_loss = 0.0
        for batch in val_loader:
        
            inputs, targets = batch[0].to(device), batch[1].to(device)
            targets = torch.permute(targets,(0,3,1,2)).float()
            inputs= torch.permute(inputs,(0,3,1,2))
            inputs = inputs.float()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
                        
            val_loss.append(loss.item())
            val_accuracy.append(accuracy(outputs,targets))

        # Calculate average validation loss and accuracy
        val_average_loss = np.mean(np.array(val_loss))
        val_average_accuracy = np.mean(np.array(val_accuracy))

        print('val_average_loss:',val_average_loss)
        print('val_average_accuracy',val_average_accuracy)

        # Log training loss and accuracy to Wandb
        wandb.log({"Validation Loss": val_average_loss, "Validation Accuracy": val_average_accuracy}, step=epoch)
       
        # print(val_average_loss)
        # Check if current model is the best model
        if val_average_loss < best_loss:
            best_loss = val_average_loss
            # Save the current best model
            torch.save(model.state_dict(), best_model_path)
        




