import torch
from matplotlib import pyplot as plt

def collate_fn(batch):
    return tuple(zip(*batch))

def time_diff(start_time, end_time):
    diff = end_time - start_time
    diff_seconds = diff.total_seconds()
    return diff_seconds

def remove_alpha_channel(image):
    if image.shape[0] == 4:
        # Remove the fourth channel (alpha channel)
        image = image[:3, :, :]
    return image

def change_extension(filename, new):
    # Split the filename into name and extension
    name, extension = filename.rsplit(".", maxsplit=1)

    # Change the extension to ".txt"
    new_filename = name + new
    return new_filename

def from_path_to_names(input_file): 
        #Example of name in input_file: "C:\validation_free\valid\060722-A-3715G-349.jpg"
        #Desired name: "060722-A-3715G-349.jpg"
        with open(input_file, "r") as f:
            file_paths = f.readlines()
            file_names = [file_path.split("\\")[-1].strip() for file_path in file_paths]

        return file_names

    
def save_loss_plot(OUT_DIR, train_loss, val_loss):
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    figure_2.savefig(f"{OUT_DIR}/valid_loss.png")
    print('SAVING PLOTS COMPLETE...')

    plt.close('all')