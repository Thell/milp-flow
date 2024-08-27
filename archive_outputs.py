import os
import datetime
import zipfile

# Get the directory of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the source directories relative to the script location
highs_output_dir = os.path.join(script_dir, "highs_output")
workerman_output_dir = os.path.join(script_dir, "workerman_output")

# Define the archive directory in the home directory
archive_dir = os.path.expanduser("~/milp-flow-output-archive")

# Create the archive directory if it doesn't exist
if not os.path.exists(archive_dir):
    os.makedirs(archive_dir)

# Generate a timestamp for organizing the archive
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a new subdirectory within the archive for this run
archive_subdir = os.path.join(archive_dir, f"archive_{timestamp}")
os.makedirs(archive_subdir)


# Function to move and zip compress files and directories recursively
def move_and_zip(source_dir, archive_subdir):
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            # Create corresponding directory in archive
            dir_path = os.path.join(root, dir_name)
            rel_dir_path = os.path.relpath(dir_path, source_dir)
            archive_dir_path = os.path.join(archive_subdir, rel_dir_path)
            os.makedirs(archive_dir_path, exist_ok=True)

        for file_name in files:
            file_path = os.path.join(root, file_name)
            rel_file_path = os.path.relpath(file_path, source_dir)
            zip_file_path = os.path.join(archive_subdir, f"{rel_file_path}.zip")
            with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_path, arcname=file_name)


# Process the highs_output directory and its subdirectories
move_and_zip(highs_output_dir, archive_subdir)

# Process the workerman_output directory and its subdirectories
move_and_zip(workerman_output_dir, archive_subdir)

print(f"Archived and compressed files to {archive_subdir}")
