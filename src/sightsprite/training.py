"""
sightsprite utilities for training data once it is acquired.  
Has utilities for labeling data, sorting it into folders, and training models. 
"""
import logging
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import shutil

logging.getLogger(__name__)

class ImageLabeler:
    """
    A minimal image labeling and review tool using matplotlib.

    This class provides a lightweight interface for labeling image datasets manually.
    Images are displayed one at a time, and the user can assign a label via keyboard
    input. Labels are saved to a CSV file and can be reviewed and modified later.

    Parameters
    ----------
    image_dir : str or Path
        Path to the directory containing images to label.
    categories : list of str
        Category names. Each is assigned to a number key (1 = categories[0], etc.).
        Supports up to 5 categories.
    output_csv : str or Path, optional
        Path to save labels assigned by user, in CSV format. 
        Default is "labels.csv". If it already exists, 
        previously labeled images will be skipped on resume. 
        CSV has two columns: "filename" and "label" 
          filename is just the image file name, not full path
          label is the category string from categories

    Public Methods
    --------------
    run()
        Launch the image labeling tool. Displays images and monitors keyboard
        inputs. If user stops in middle of labeling dataset, it will
        automatically resume from where labeling last stopped if 
        output_csv exists.

    review_labels()
        Launch the label review tool. If output_csv exists, will cycle
        through existing labels, allowing the user to relabel or delete
        labels saved in the CSV file.
    """
    def __init__(self, image_dir, categories, output_csv="labels.csv"):
        if len(categories) > 5:
            raise ValueError("This minimal tool only supports up to 5 categories.")

        self.image_dir = Path(image_dir)
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        self.categories = categories
        # map categories to keyboard numbers for labeling 
        self.category_keys = {str(i + 1) for i in range(len(self.categories))}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

        self.image_paths = self._load_image_paths() #  all images in image_dir
        # list of (filename, label) tuples to save (saves every 10, or on quit)
        self.labels = [] 
        self.current_index = 0
        self.fig = None
        self.ax = None
        self.fontsize = 10

    def run(self):
        """
        Launch the main image labeling tool. This is the 
        main entry point for the user, the main point of 
        this class. 

        Displays images for labeling, monitors keyboard inputs.
        """
        if not self.image_paths:
            print("No valid images found or all images labeled.")
            return

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.canvas.manager.set_window_title("Image Labeling Tool")
        # connect key press event to custom handler method 
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._update_display()
        plt.show()

    def _load_image_paths(self):
        """
        Return a list of image paths to label, skipping those already labeled.
        """
        all_paths = self._get_all_image_paths()
        labeled = self._get_labeled_filenames()
        return [fpath for fpath in all_paths if fpath.name not in labeled]

    def _get_all_image_paths(self):
        """
        Get all valid image files in image_dir based on allowed extensions.
        """
        filenames = sorted(os.listdir(self.image_dir))
        return [self.image_dir / fname for fname in filenames
                if (self.image_dir / fname).is_file()
                and (self.image_dir / fname).suffix.lower() in self.image_extensions]

    def _get_labeled_filenames(self):
        """
        Return a set of filenames already labeled in the 
        output CSV, if it exists.
        """
        if not self.output_csv.exists():
            return set()
        try:
            df = pd.read_csv(self.output_csv)
            print(f"Resuming: Found {len(df)} labeled images.")
            return set(df["filename"].tolist())
        except Exception as e:
            print(f"Failed to read CSV. Starting fresh. Error: {e}")
            return set()
    
    def _save_labels(self, force=False):
        """
        Check if list of label tuples has passed threshold for saving.
        If so, create df and save to CSV, and then clear tuples list. 
        More efficient than saving to CSV after every label. If force is 
        True, it will save (this is used when quitting early). 
        """
        if len(self.labels) >= 10 or force:
            if self.labels:
                new_df = pd.DataFrame(self.labels, columns=["filename", "label"])
                try:
                    if self.output_csv.exists():
                        existing_df = pd.read_csv(self.output_csv)
                        df = pd.concat([existing_df, new_df], ignore_index=True)
                    else:
                        df = new_df
                    df.to_csv(self.output_csv, index=False)
                    print(f"Saved {len(self.labels)} labels to {self.output_csv}")
                    # clear labels after saving -- starting fresh
                    self.labels.clear()
                except Exception as e:
                    print(f"Failed to save CSV. Labels kept in memory. Error: {e}")

    def _update_display(self):
        self.ax.clear()
        try:
            img = Image.open(self.image_paths[self.current_index])
            self.ax.imshow(img)

            line1 = f"{self.image_paths[self.current_index].name} ({self.current_index + 1}/{len(self.image_paths)})"
            line2 = " | ".join([f"{i+1} = {cat}" for i, cat in enumerate(self.categories)])
            line3 = "n = next | q = quit"
            full_title = f"{line1}\n{line2}\n{line3}"
            self.ax.set_title(full_title, fontsize=self.fontsize)

            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.fig.canvas.draw()
            self.fig.subplots_adjust(top=0.85)
        except Exception as e:
            print(f"Failed to load {self.image_paths[self.current_index]}: {e}")
            self.current_index += 1
            if self.current_index < len(self.image_paths):
                self._update_display()
            else:
                self._save_labels(force=True)
                plt.close(self.fig)

    def _on_key(self, event):
        if event.key == "q":
            print("Quitting. Saving labels...")
            self._save_labels(force=True)
            plt.close(self.fig)
            return

        if event.key == "n":
            print(f"Skipped: {self.image_paths[self.current_index].name}")
            self.current_index += 1

        elif event.key in self.category_keys:
            self._label_current_image(event.key)

        else:
            print(f"Ignored key: {event.key}")
            return

        # Once key event processed: update display or finish if done
        if self.current_index < len(self.image_paths):
            self._update_display()
        else:
            print("Finished labeling all images.")
            self._save_labels(force=True)
            plt.close(self.fig)

    def _label_current_image(self, key):
        """
        Label current image with category depending on the user key pressed. 
        Key will be a number (as string), e.g. '1' for first category.
        will be number between 1 and len(categories).
        """
        label_index = int(key) - 1  #- 1 b/c categories are 1-based for user
        label = self.categories[label_index] # convert to string-based label
        filename = self.image_paths[self.current_index].name # just filename, not fully path

        # Add label and attempt batch save
        self.labels.append((filename, label))
        print(f"Labeled: {filename} -> {label}")
        self._save_labels()

        self.current_index += 1

    def review_labels(self):
        """
        The second main entry point for the user.
        Launches the label review tool. Lets user navigate previously
        labeled data, showing existing labels along with the image. 
        Allows the user to relabel image, delete label, or keep 
        things the same. 
        """
        if not self.output_csv.exists():
            print(f"No labels found at {self.output_csv}.")
            return

        try:
            df = pd.read_csv(self.output_csv)
            original_labels = list(zip(df["filename"], df["label"]))
            if not original_labels:
                print("No labeled images in CSV.")
                return
            # print how many assigned to each category 
            print("Label distribution:")
            print(df["label"].value_counts().to_string())
        except Exception as e:
            print(f"Failed to read CSV: {e}")
            return
        # index of current image being reviewed
        self.review_index = 0
        # tuple of labels for review 
        self.review_labels = original_labels
        self.review_df = df
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.canvas.manager.set_window_title("Label Review Tool")
        # same logic as in run() but for review tool 
        self.fig.canvas.mpl_connect("key_press_event", self._on_review_key)
        self._update_review_display()
        plt.show()

    def _on_review_key(self, event):
        if event.key == "q":
            print("Quitting review.")
            plt.close(self.fig)
            return
        elif event.key == "right":
            self.review_index = min(self.review_index + 1, len(self.review_labels) - 1)
        elif event.key == "left":
            self.review_index = max(self.review_index - 1, 0)
        elif event.key == "d":
            self._delete_current_label()
        elif event.key in self.category_keys:
            self._relabel_current_image(event.key)
        else:
            print(f"Ignored key: {event.key}")
            return

        self._update_review_display()

    def _relabel_current_image(self, key):
        new_label = self.categories[int(key) - 1]
        filename = self.review_labels[self.review_index][0]
        original_label = self.review_df.at[self.review_index, "label"]

        if new_label != original_label:
            print(f"Relabeling {filename} to {new_label}")
            self.review_df.at[self.review_index, "label"] = new_label
            self.review_df.to_csv(self.output_csv, index=False)
            self.review_labels[self.review_index] = (filename, new_label)
        else:
            print(f"No change: {filename} remains labeled as {original_label}")

    def _delete_current_label(self):
        """
        Delete the current label from the review set (both memory and disk).
        Closes the viewer if no labels remain.
        """
        filename = self.review_labels[self.review_index][0]
        print(f"Removing label for {filename}")

        # Drop from dataframe and save to csv, and update review_labels
        self.review_df.drop(self.review_index, inplace=True)
        self.review_df.to_csv(self.output_csv, index=False)
        self.review_labels.pop(self.review_index)

        # If nothing left, exit
        if not self.review_labels:
            print("No more labeled images to review.")
            plt.close(self.fig)
            return

        # If we deleted last item, ensure index stays in bounds
        if self.review_index >= len(self.review_labels):
            self.review_index = len(self.review_labels) - 1

    def _update_review_display(self):
        self.ax.clear()
        try:
            filename, label = self.review_labels[self.review_index]
            img = Image.open(self.image_dir / filename)
            self.ax.imshow(img)

            line1 = f"({filename}) Labeled {label} ({self.review_index + 1}/{len(self.review_labels)})"
            relabel_options = [f"Change to: {i+1} = {cat}" for i, cat in enumerate(self.categories) if cat != label]
            line2 = " | ".join(relabel_options) + " | d = delete"
            line3 = "left/right = navigate | q = quit"
            full_title = f"{line1}\n{line2}\n{line3}"
            self.ax.set_title(full_title, fontsize=self.fontsize)

            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.fig.canvas.draw()
            self.fig.subplots_adjust(top=0.85)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            self.review_index += 1
            if self.review_index < len(self.review_labels):
                self._update_review_display()
            else:
                print("No more labeled images to review.")
                plt.close(self.fig)


def sort_images_by_label(labels_file, source_dir, output_dir):
    """
    Organize images into folders based on their labels stored in a CSV file.

    Moves each image from the source directory into output directory 
    under a subdirectory named after its label. The labels are read from a
    CSV file with two columns: 'filename' and 'label'. 

    Parameters
    ----------
    labels_file : str or Path
        Path to the CSV file with image labels. Must contain 
        columns 'filename' and 'label'.
    source_dir : str or Path
        Directory containing the original labeled images.
    output_dir : str or Path
        Directory where the reorganized image folders will be created.

    Returns
    -------
    None
        Images are copied to `output_dir/label/filename`.

    Notes
    -----
    - Label directories are created based on sorted label names to ensure deterministic ordering.
    - This avoids downstream issues with PyTorch's ImageFolder, which sorts subdirectories 
      to assign class indices.
    - Files are copied, not moved, so the original images remain unchanged.
    """
    logging.info(f"Sorting images from {source_dir} to {output_dir}.")
    logging.info(f"Is using labels in {labels_file}")

    df = pd.read_csv(labels_file)
    # check that required columns exist
    if not {'filename', 'label'}.issubset(df.columns):
        raise ValueError("labels_file must contain 'filename' and 'label' columns.")
    
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Sort labels to ensure consistent folder creation order
    labels_sorted = sorted(df["label"].unique())

    # Handle images for each label separately
    for label in labels_sorted:
        label_df = df[df["label"] == label]
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for _, row in label_df.iterrows():
            filename = row["filename"]
            source = source_dir / filename
            destination = label_dir / filename
            shutil.copy2(source, destination)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logging.info("__TESTING SIGHTSPRITE TRAINING__")

    data_dir = Path(r'C:/Users/Eric/development/data/')
    output_dir = data_dir / "pets_random"
    pets_dir = data_dir / "pets"

    # test_class, sort_images
    test_option = "test_class" 

    if test_option == "test_class":
        app_home = Path.home() / ".sightsprite"
        save_dir = app_home / "labels.csv"
        categories = ["dog", "cat"]
        labeler = ImageLabeler(output_dir, categories, output_csv=save_dir)
        labeler.run()
        # labeler.review_labels()

    elif test_option == "sort_images":
        app_home = Path.home() / ".sightsprite"
        labels_file = app_home / "labels.csv"
        source_dir = output_dir
        output_dir = data_dir / "sorted_pets"
        sort_images_by_label(labels_file, source_dir, output_dir)

        logging.info(f"Sorting done. Check {output_dir} for results.")

