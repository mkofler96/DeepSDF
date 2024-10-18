import os
import shutil
import argparse

def convert_size(size_bytes):
    """Convert size in bytes to a human-readable format (KB, MB, GB, etc.)"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(min(len(size_name) - 1, (size_bytes.bit_length() - 1) // 10))
    p = 1 << (i * 10)
    return f"{size_bytes / p:.2f} {size_name[i]}"

def find_and_copy_files(source_dir, destination_dir):
    # Check if destination directory exists, if not, create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Created directory: {destination_dir}")

    total_size = 0  # Variable to accumulate the total size of copied files

    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check if the file contains 'surf' and has a '.inp' extension
            if 'surf' in file and file.endswith('.inp'):
                # Construct full file path
                full_file_path = os.path.join(root, file)
                
                # Get the file size and add it to the cumulative total
                file_size = os.path.getsize(full_file_path)
                total_size += file_size

                # Copy the file to the destination directory
                shutil.copy(full_file_path, destination_dir)
                print(f"Copied: {full_file_path} ({convert_size(file_size)}) to {destination_dir}")

    # Print total size of all copied files in human-readable format
    print(f"\nTotal size of copied files: {convert_size(total_size)}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Find and copy *surf*.inp files to a new directory.")
    parser.add_argument("source_dir", help="The source directory to search for files")
    parser.add_argument("destination_dir", help="The destination directory to copy the files")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to find and copy the files
    find_and_copy_files(args.source_dir, args.destination_dir)
