import os
from PIL import Image
import glob


def crop_images(input_folder, output_folder, crop_size=640):
    """
    Crops images from input_folder into non-overlapping squares of crop_size x crop_size
    and saves them to output_folder. Discards partial crops at edges.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, extension)))

    if not image_files:
        print(f"No image files found in {input_folder}")
        return

    total_crops = 0
    for image_file in image_files:
        try:
            # Open the image
            img = Image.open(image_file)
            width, height = img.size

            # Get image basename for naming crops
            base_name = os.path.splitext(os.path.basename(image_file))[0]

            # Calculate number of full crops in each dimension
            num_crops_width = width // crop_size
            num_crops_height = height // crop_size

            print(f"Processing {image_file} ({width}x{height}) -> {num_crops_width}x{num_crops_height} crops")

            crop_count = 0
            # Extract crops
            for i in range(num_crops_width):
                for j in range(num_crops_height):
                    left = i * crop_size
                    upper = j * crop_size
                    right = left + crop_size
                    lower = upper + crop_size

                    crop = img.crop((left, upper, right, lower))

                    # Save the crop
                    crop_filename = f"{base_name}_crop_{i}_{j}.jpg"
                    crop.save(os.path.join(output_folder, crop_filename))
                    crop_count += 1

            print(f"Generated {crop_count} crops from {image_file}")
            total_crops += crop_count

        except Exception as e:
            print(f"Error processing {image_file}: {e}")

    print(f"Total crops generated: {total_crops}")


if __name__ == "__main__":
    # Default folder paths - modify these as needed
    default_input_folder = "Google_Images"
    default_output_folder = "cropped_images"
    default_crop_size = 640

    # Ask user if they want to use custom paths or defaults
    print(f"Default input folder: {default_input_folder}")
    custom_input = input("Enter custom input folder path or press Enter to use default: ")
    input_folder = custom_input if custom_input else default_input_folder

    print(f"Default output folder: {default_output_folder}")
    custom_output = input("Enter custom output folder path or press Enter to use default: ")
    output_folder = custom_output if custom_output else default_output_folder

    print(f"Default crop size: {default_crop_size}x{default_crop_size}")
    custom_size = input("Enter custom crop size or press Enter to use default: ")
    crop_size = int(custom_size) if custom_size.isdigit() else default_crop_size

    print(f"\nProcessing with these settings:")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Crop size: {crop_size}x{crop_size}")

    # Run the cropping function
    crop_images(input_folder, output_folder, crop_size)