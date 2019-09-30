import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import foundations
matplotlib.use("Agg")


def calc_dice(target, prediction, smooth=0.01):
    """
    Sorenson Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth (target) mask and P is the prediction mask
    """
    prediction = np.round(prediction)

    numerator = 2.0 * np.sum(target * prediction) + smooth
    denominator = np.sum(target) + np.sum(prediction) + smooth
    coef = numerator / denominator

    return coef


def calc_soft_dice(target, prediction, smooth=0.01):
    """
    Sorensen (Soft) Dice coefficient - Don't round preictions
    """
    numerator = 2.0 * np.sum(target * prediction) + smooth
    denominator = np.sum(target) + np.sum(prediction) + smooth
    coef = numerator / denominator

    return coef


def plot_results(model, imgs_validation, msks_validation,
                 img_no, png_directory):
    """
    Calculate the Dice and plot the predicted masks for image # img_no
    """

    img = imgs_validation[img_no: img_no+1]
    msk = msks_validation[img_no: img_no+1]

    pred_mask = model.predict(img)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img[0, :, :, 0], cmap="bone", origin="lower")
    plt.title("MRI")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(msk[0, :, :, 0], origin="lower")
    plt.title("Ground Truth")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask[0, :, :, 0], origin="lower")
    plt.title("Prediction\n(Dice = {:.4f})".format(calc_dice(msk, pred_mask)))
    plt.axis("off")

    png_filename = os.path.join(png_directory, "pred_{}.png".format(img_no))
    plt.savefig(png_filename, bbox_inches="tight", pad_inches=0)

    foundations.save_artifact(png_filename)
    
    print("Dice {:.4f}, Soft Dice {:.4f}, Saved png file to: {}".format(
        calc_dice(msk, pred_mask), calc_soft_dice(msk, pred_mask), png_filename))

