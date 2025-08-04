#
# Title: Poultry Vocalization Spectrogram Viewer
#
# Description: This script iterates through all the audio files in the "SmartEars"
# dataset, generates a spectrogram for each file, and saves it as a PNG image.
# The images will be saved into subdirectories corresponding to their category
# ('Healthy', 'Sick', 'None') inside a main 'spectrogram_images' folder.
#

#
# I. Setup and Installation
#
# In this section, we install the necessary R packages for audio processing
# and plotting.
#

# --- One-time Setup ---
# install.packages(c("tuneR", "seewave"))
# --------------------

# Load the required libraries for the session
library(tuneR)
library(seewave)

#
# II. Set Data Path
#
# This section defines the location of your dataset. Please ensure the path
# below points to the main folder containing the 'Healthy', 'Sick', and 'None'
# subfolders.
#

DATA_PATH <- 'C:/Users/srini/OneDrive/Desktop/R/Omen-rstudio/SmartEars A Practical Framework for Poultry Respiratory Monitoring via Spectrogram-Based Audio Classification and AI-Assisted Labeling'

# Check if the data directory exists
if (!dir.exists(DATA_PATH)) {
  stop(paste("Dataset directory not found at:", DATA_PATH, ". Please ensure the path is correct."))
} else {
  print("Dataset directory found. Proceeding with visualization.")
}


#
# III. Load File List & Create Output Directories
#
# Get a list of all .wav files and set up the folders to save the images.
#

print("Scanning for audio files...")
file_list <- list.files(DATA_PATH, recursive = TRUE, full.names = TRUE, pattern = "\\.wav$")

if (length(file_list) == 0) {
  stop("No .wav files found in the specified directory. Please check the DATA_PATH.")
}

# Create a main directory for the output images
output_dir <- "spectrogram_images"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

print(paste("Found", length(file_list), "audio files. Starting spectrogram generation..."))
print(paste("Images will be saved in the '", output_dir, "' folder.", sep=""))


#
# IV. Generate and Save Spectrograms
#
# This loop will go through each file, generate its spectrogram, and save it as a PNG.
#

for (i in 1:length(file_list)) {
  file_path <- file_list[i]
  
  tryCatch({
    # Get the category to create a subdirectory
    category <- basename(dirname(file_path))
    category_dir <- file.path(output_dir, category)
    if (!dir.exists(category_dir)) {
      dir.create(category_dir)
    }
    
    # Create the output filename by replacing .wav with .png
    png_filename <- sub("\\.wav$", ".png", basename(file_path))
    output_path <- file.path(category_dir, png_filename)
    
    # Read the audio file
    wave <- tuneR::readWave(file_path)
    
    # Get the filename for the plot title
    filename <- basename(file_path)
    plot_title <- paste("Category:", category, "\nFile:", filename)
    
    # Open a PNG graphics device to save the plot to a file
    png(filename = output_path, width = 800, height = 600)
    
    # Generate the spectrogram (this will be drawn on the PNG device)
    seewave::spectro(wave, wl = 512, ovlp = 75, main = plot_title, xlab = "Time (s)", ylab = "Frequency (kHz)")
    
    # Close the graphics device, which finalizes and saves the file
    dev.off()
    
    # Print progress to the console
    print(paste("(", i, "/", length(file_list), ") Saved spectrogram for:", filename))
    
  }, error = function(e) {
    # If an error occurs with a file, print it and move to the next one
    print(paste("Could not process file:", file_path, "\nError:", e$message))
  })
}

print("Finished generating all spectrograms.")
