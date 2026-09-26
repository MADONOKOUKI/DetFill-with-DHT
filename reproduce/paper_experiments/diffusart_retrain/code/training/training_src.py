import torch
import os
from torch.optim import optimizer, Adam
from torchvision.utils import save_image
from pathlib import Path
from utils import *
from training.losses import *
from training.forward import *
from torch.utils.tensorboard import SummaryWriter
import time


# Checkpoint dir
sub_dir = 'test_test_test'
checkpoint_dir = ('./checkpoint/' + sub_dir)
checkpoint_dir1 = ('./checkpoint/' + sub_dir + '_encd')
model_dir = ('./save_models/' + sub_dir)
model_dir1 = ('./save_models/' + sub_dir + '_encd')
PATH = './net_start.pth'
PATH_enc = './net_encd.pth'
# Save images in tensorboard
save_val_img = 200
save_train_img = 4500
save_model_train = 4500


# results_folder = Path("./results")
# results_folder.mkdir(exist_ok=True)
epochs = 200


loss_val_epoch = 0.0
running_loss = 0.0



def training_dif(model, optimizer, dataloader, dataloader_test, timesteps, channels, image_size, device):
    # Writer will output to ./runs/ directory by default
    tb = SummaryWriter('./runs/' + sub_dir + '/', flush_secs=1)
    # model.train()
    for epoch in range(epochs):
        model.train()
        if epoch == 0:
            best_loss = 1000.0
            sum_idx = 0
            running_loss = 0.0
        else:
            sum_idx += len(dataloader)
            running_loss = 0.0

        for idx, (original, sketch_data) in enumerate(dataloader):
            step_start_time = time.time()
            optimizer.zero_grad()

            batch_size = original.shape[0]
            original = original.to(device, dtype=torch.float32)
            sketch_data = sketch_data.to(device, dtype=torch.float32)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()


            loss = p_losses(model, original, sketch_data, t, loss_type="l2")

            if idx % 2 == 0:
                print(
                    "Epoch[{}/{}]({}/{}): Loss_train: {:.4f}, Step time: {} seconds".format(
                        epoch, epochs, idx, len(dataloader),
                        loss.item(), time.time() - step_start_time))

            running_loss = running_loss + loss

            loss.backward()
            optimizer.step()

            if (idx + sum_idx > 10000) and (loss.item() < best_loss):
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                is_best = True
                best_loss = loss.item()


                save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)
                print('Best model updated')

            if (idx % save_model_train == 0):
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                is_best = True
                save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)
                print('Last model saved', is_best)

            # Write some images in tensorboard every save_train_img step
            if (idx % save_train_img == 0):
                samples = sample(model, sketch_data[0:2, :, :, :], image_size=image_size, batch_size=2, channels=channels)
                img_grid = torchvision.utils.make_grid(deprocess(original))
                sketch_grid = torchvision.utils.make_grid(deprocess(sketch_data))
                samples_grid = torchvision.utils.make_grid(deprocess(samples[9]))
                samples_grid_150 = torchvision.utils.make_grid(deprocess(samples[14]))
                samples_grid_1000 = torchvision.utils.make_grid(deprocess(samples[8]))
                tb.add_image('Original in step', img_grid, idx + sum_idx)
                tb.add_image('Sketch image in step', sketch_grid, idx + sum_idx)
                tb.add_image('Samples 10 in step', samples_grid, idx + sum_idx)
                tb.add_image('Samples 150 in step', samples_grid_150, idx + sum_idx)
                tb.add_image('Samples 1000 in step', samples_grid_1000, idx + sum_idx)
                tb.flush()

            # Writing losses in tensorboard
            tb.add_scalar("Total loss per step", loss, idx + sum_idx)
            # Forcing refreshing on tensorboard
            tb.flush()

        # Average loss for each epoch

        epoch_loss = running_loss / len(dataloader)

        tb.add_scalar("Total loss per epoch", epoch_loss, epoch)

        # Save final model

    torch.save(model.state_dict(), PATH)
    print('Training saved and finished')
    tb.close()
