import matplotlib.pyplot as plt


def plot_images_list(images, figsize=(20, 10), axis=False):
    fig, ax = plt.subplots(ncols=len(images), figsize=figsize)
    for index, image in enumerate(images):
        ax[index].imshow(image)

    if axis == False:
        disable_axis(ax)

    return fig

def disable_axis(ax):
    for a in ax:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
    plt.axis('off')
