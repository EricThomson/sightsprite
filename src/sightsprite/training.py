"""
sightsprite utilities for training once data is acquired using capture module.  
Has utilities for labeling data, sorting it into folders, and training models. 
"""
import logging
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from PIL import Image
from PIL import ImageEnhance
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
        self.brightness = 1.0 
        # map categories to keyboard numbers for labeling 
        self.category_keys = {str(i + 1) for i in range(len(self.categories))}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

        self.image_paths = self._load_image_paths() # all images in image_dir
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
            logging.warning("No valid images found or all images labeled.")
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
            labeled_filenames = set(df["filename"].tolist())
            num_remaining = len(self._get_all_image_paths()) - len(labeled_filenames)
            logging.info(f"Resuming: Found {len(labeled_filenames)} labeled images. {num_remaining} images left to label.")
            return labeled_filenames
        except Exception as e:
            logging.warning(f"Failed to read CSV. Starting fresh. Error: {e}")
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
                    logging.info(f"Saved {len(self.labels)} labels to {self.output_csv}")
                    # clear labels after saving -- starting fresh
                    self.labels.clear()
                except Exception as e:
                    logging.warning(f"Failed to save CSV. Labels kept in memory. Error: {e}")

    def _update_display(self, maintain_zoom=False):
        # if we maintaining zoom of current axes, save current limits
        if maintain_zoom:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

        self.ax.clear()
        try:
            img = Image.open(self.image_paths[self.current_index])
            img = self._apply_brightness(img)
            self.ax.imshow(img)

            # line 1 depends on whether current image is labeled
            filename = self.image_paths[self.current_index].name
            label = self._get_label_for_image(filename)
            if label:
                line1 = f"({filename}) Labeled {label} ({self.current_index + 1}/{len(self.image_paths)})"
            else:
                line1 = f"({filename}) UNLABELED ({self.current_index + 1}/{len(self.image_paths)})"
            # line 2 shows category options
            line2 = " | ".join([f"{i+1} = {cat}" for i, cat in enumerate(self.categories)])
            # line 3 contains instructions
            line3 = "left/right: next/prev | up/down = brightness | q = quit"
            full_title = f"{line1}\n{line2}\n{line3}"
            self.ax.set_title(full_title, fontsize=self.fontsize)

            self.ax.set_xticks([])
            self.ax.set_yticks([])

            if maintain_zoom:
                # Restore saved zoom limits
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)

            self.fig.canvas.draw()
            self.fig.subplots_adjust(top=0.85)


        except Exception as e:
            logging.warning(f"Failed to load {self.image_paths[self.current_index]}: {e}")
            self.current_index += 1
            if self.current_index < len(self.image_paths):
                self._update_display()
            else:
                self._save_labels(force=True)
                plt.close(self.fig)


    def _apply_brightness(self, img):
        """
        Apply brightness adjustment to a PIL Image using the current brightness.

        Parameters
        ----------
        img : PIL.Image
            Original image.

        Returns
        -------
        PIL.Image
            Brightness-adjusted image.
        """
        if self.brightness == 1.0:
            return img
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(self.brightness)


    def _get_label_for_image(self, filename):
        """
        Return the label for an image if it exists in memory or CSV.
        Otherwise returns None.
        """
        # Check in-memory labels first
        for fname, lbl in self.labels:
            if fname == filename:
                return lbl

        # Check CSV
        if self.output_csv.exists():
            try:
                df = pd.read_csv(self.output_csv)
                mask = df["filename"] == filename
                if mask.any():
                    return df.loc[mask, "label"].values[0]
            except Exception as e:
                logging.warning(f"Error checking label for {filename}: {e}")

        return None

    def _on_key(self, event):
        if event.key == "q":
            logging.info("Quitting. Saving labels...")
            self._save_labels(force=True)
            plt.close(self.fig)
            return

        maintain_zoom = False # default, only deviate if brightness changed
        if event.key == "right":
            self._go_forward_one_image()

        elif event.key == "left":
            self._go_back_one_image()

        elif event.key in self.category_keys:
            self._label_current_image(event.key)

        elif event.key == "up":
            self.brightness *= 1.1
            logging.info(f"Brightness increased to {self.brightness:.1f}")
            maintain_zoom = True

        elif event.key == "down":
            self.brightness /= 1.1
            logging.info(f"Brightness decreased to {self.brightness:.1f}")
            maintain_zoom = True
        
        else:
            logging.info(f"Ignored key: {event.key}")
            return

        # Once key event processed: update display or finish if done
        if self.current_index < len(self.image_paths):
            self._update_display(maintain_zoom=maintain_zoom)
        else:
            logging.info("Finished labeling all images.")
            self._save_labels(force=True)
            plt.close(self.fig)

    def _go_forward_one_image(self):
        """
        Handle indexing changes if user clicks next image in list of images, 
        if user clicks right arrow. 
        """
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            logging.info(f"Moved forward to: {self.image_paths[self.current_index].name}")
        else:
            logging.info("Already at the last image.")

    def _go_back_one_image(self):
        """
        Handle going back one image. 
        Only navigates back; does not delete any labels.
        """
        if self.current_index > 0:
            self.current_index -= 1
            logging.info(f"Moved back to: {self.image_paths[self.current_index].name}")
        else:
            logging.info("Already at the first image. Cannot go back further.")

    def _label_current_image(self, key):
        """
        Label current image with category depending on the user key pressed. 
        Key will be a number (as string), e.g. '1' for first category.
        will be number between 1 and len(categories).
        """
        label_index = int(key) - 1
        label = self.categories[label_index]
        filename = self.image_paths[self.current_index].name

        # Remove previous label for this image if it exists
        self.labels = [
            (fname, lbl)
            for fname, lbl in self.labels
            if fname != filename
        ]

        self.labels.append((filename, label))
        logging.info(f"Labeled: {filename} -> {label}")
        self._save_labels()

        self.current_index += 1

    # review_labels() and related methods unchanged from canonical code
    def review_labels(self):
        """
        The second main entry point for the user.
        Launches the label review tool. Lets user navigate previously
        labeled data, showing existing labels along with the image. 
        Allows the user to relabel image, delete label, or keep 
        things the same. 
        """
        if not self.output_csv.exists():
            logging.warning(f"No labels found at {self.output_csv}.")
            return

        try:
            df = pd.read_csv(self.output_csv)
            original_labels = list(zip(df["filename"], df["label"]))
            if not original_labels:
                logging.warning("No labeled images in CSV.")
                return
            print("Label distribution:")
            print(df["label"].value_counts().to_string())
        except Exception as e:
            logging.warning(f"Failed to read CSV: {e}")
            return
        self.review_index = 0
        self.review_labels = original_labels
        self.review_df = df
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.canvas.manager.set_window_title("Label Review Tool")
        self.fig.canvas.mpl_connect("key_press_event", self._on_review_key)
        self._update_review_display()
        plt.show()

    def _on_review_key(self, event):
        if event.key == "q":
            logging.info("Quitting review.")
            plt.close(self.fig)
            return

        maintain_zoom = False  # default

        if event.key == "right":
            self.review_index = min(self.review_index + 1, len(self.review_labels) - 1)

        elif event.key == "left":
            self.review_index = max(self.review_index - 1, 0)

        elif event.key == "d":
            self._delete_current_label()

        elif event.key in self.category_keys:
            self._relabel_current_image(event.key)

        elif event.key == "up":
            self.brightness *= 1.1
            logging.info(f"Brightness increased to {self.brightness:.1f}")
            maintain_zoom = True

        elif event.key == "down":
            self.brightness /= 1.1
            logging.info(f"Brightness decreased to {self.brightness:.1f}")
            maintain_zoom = True

        else:
            logging.info(f"Ignored key: {event.key}")
            return

        self._update_review_display(maintain_zoom=maintain_zoom)

    def _relabel_current_image(self, key):
        new_label = self.categories[int(key) - 1]
        filename = self.review_labels[self.review_index][0]
        original_label = self.review_df.at[self.review_index, "label"]

        if new_label != original_label:
            logging.info(f"Relabeling {filename} to {new_label}")
            self.review_df.at[self.review_index, "label"] = new_label
            self.review_df.to_csv(self.output_csv, index=False)
            self.review_labels[self.review_index] = (filename, new_label)
        else:
            logging.info(f"No change: {filename} remains labeled as {original_label}")

    def _delete_current_label(self):
        """
        Delete the current label from the review set (both memory and disk).
        Closes the viewer if no labels remain.
        """
        filename = self.review_labels[self.review_index][0]
        logging.info(f"Removing label for {filename}")

        self.review_df.drop(self.review_index, inplace=True)
        self.review_df.to_csv(self.output_csv, index=False)
        self.review_labels.pop(self.review_index)

        if not self.review_labels:
            logging.info("No more labeled images to review.")
            plt.close(self.fig)
            return

        if self.review_index >= len(self.review_labels):
            self.review_index = len(self.review_labels) - 1

    def _update_review_display(self, maintain_zoom=False):

        if maintain_zoom:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

        self.ax.clear()
        try:
            filename, label = self.review_labels[self.review_index]
            img = Image.open(self.image_dir / filename)
            img = self._apply_brightness(img)
            self.ax.imshow(img)

            line1 = f"({filename}) Labeled {label} ({self.review_index + 1}/{len(self.review_labels)})"
            relabel_options = [f"Change to: {i+1} = {cat}" for i, cat in enumerate(self.categories) if cat != label]
            line2 = " | ".join(relabel_options) + " | d = delete"
            line3 = "left/right: next/prev | up/down = brightness | q = quit"
            full_title = f"{line1}\n{line2}\n{line3}"
            self.ax.set_title(full_title, fontsize=self.fontsize)

            self.ax.set_xticks([])
            self.ax.set_yticks([])
            if maintain_zoom:
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)

            self.fig.canvas.draw()
            self.fig.subplots_adjust(top=0.85)
        except Exception as e:
            logging.warning(f"Failed to load {filename}: {e}")
            self.review_index += 1
            if self.review_index < len(self.review_labels):
                self._update_review_display()
            else:
                logging.info("No more labeled images to review.")
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

    data_dir = Path(__file__).parent / "data"
    pets_data = data_dir / "pets" # to-be-labeled images, packaged with sightsprite
    app_home = Path.home() / ".sightsprite"
    labels_file = app_home / "pet_labels.csv"
    pets_categorized = app_home / "pets_sorted" # categorized images in cat/ dog/ folders 
    pet_categories = ["dog", "cat"] 

    # Options: test_label, test_review, sort_images
    test_option = "test_label" 

    if test_option == "test_label":
        # ImageLabeler(data_dir, categories, output_csv_path) 
        labeler = ImageLabeler(pets_data, pet_categories, output_csv=labels_file)
        labeler.run()

    elif test_option == "test_review":
        # review previously labeled images
        labeler = ImageLabeler(pets_data, pet_categories, output_csv=labels_file)
        labeler.review_labels()

    elif test_option == "sort_images":
        sort_images_by_label(labels_file, pets_data, pets_categorized)

        logging.info(f"Sorting done. Check {pets_categorized} for results.")

