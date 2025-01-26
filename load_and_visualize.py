import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

DATASET_PATH = "../data/train/images"  

def load_image(file_name):
    """Load an image from the dataset."""
    image_path = os.path.join(DATASET_PATH, file_name)
    image = cv2.imread(image_path) 
    if image is None:
        raise FileNotFoundError(f"Image {file_name} not found in {DATASET_PATH}")
    return image

def display_image_with_matplotlib(image, title="Image"):
    """Display the image using Matplotlib."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")  
    plt.show()

def split_and_analyze_channels(image):
    """Split the image into its color channels and display numeric statistics."""
    b, g, r = cv2.split(image)  
    stats = {
        "Blue": {
            "Min": np.min(b),
            "Max": np.max(b),
            "Mean": np.mean(b),
            "StdDev": np.std(b),
        },
        "Green": {
            "Min": np.min(g),
            "Max": np.max(g),
            "Mean": np.mean(g),
            "StdDev": np.std(g),
        },
        "Red": {
            "Min": np.min(r),
            "Max": np.max(r),
            "Mean": np.mean(r),
            "StdDev": np.std(r),
        },
    }

    for channel, stat in stats.items():
        print(f"\n{channel} Channel Statistics:")
        for key, value in stat.items():
            print(f"  {key}: {value:.2f}")

    return stats

def plot_histogram(image):
    """Plot the histogram of pixel intensities for each channel."""
    colors = ('b', 'g', 'r')  
    histograms = {}
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        histograms[color] = hist.flatten()  
        plt.plot(hist, color=color)
    plt.xlim([0, 256])
    plt.title("Histogram of Pixel Intensities")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend(["Blue", "Green", "Red"])
    plt.show()
    for color, hist in histograms.items():
        print(f"\n{color.upper()} Channel Histogram Summary:")
        print(f"  Most Frequent Intensity: {np.argmax(hist)}")
        print(f"  Frequency: {np.max(hist):.0f}")
        print(f"  Total Pixel Count: {np.sum(hist):.0f}")

if __name__ == "__main__":
    sample_file = "IMG_2274_jpeg_jpg.rf.2f319e949748145fb22dcb52bb325a0c.jpg"  
    image = load_image(sample_file)
    display_image_with_matplotlib(image, title="Sample Image - Matplotlib")
    stats = split_and_analyze_channels(image)
    plot_histogram(image)
