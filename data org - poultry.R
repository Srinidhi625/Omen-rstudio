# --- PREPARATORY SCRIPT: ORGANIZE AND RENAME DATA ---

# 1. DEFINE YOUR PATHS
# Path to the root of the downloaded and unzipped Mendeley dataset
source_data_root <- "C:/Users/srini/OneDrive/Poultry/SmartEars A Practical Framework for Poultry Respiratory Monitoring via Spectrogram-Based Audio Classification and AI-Assisted Labeling"

# Path to the CSV file with labels
labels_csv_path <- file.path(source_data_root, "infection_labels.csv")

# The directory where you want to create your sorted 'data' folder for the CNN
output_root <- getwd() # Uses your current R working directory
data_dir_output <- file.path(output_root, "data")
sick_dir_output <- file.path(data_dir_output, "sick")
healthy_dir_output <- file.path(data_dir_output, "healthy")

# 2. CREATE OUTPUT DIRECTORIES
if (!dir.exists(data_dir_output)) dir.create(data_dir_output)
if (!dir.exists(sick_dir_output)) dir.create(sick_dir_output)
if (!dir.exists(healthy_dir_output)) dir.create(healthy_dir_output)

# 3. READ THE LABELS
if (!file.exists(labels_csv_path)) {
  stop("Error: infection_labels.csv not found at the specified path: ", labels_csv_path)
}
labels_df <- read.csv(labels_csv_path)

# 4. PROCESS AND COPY FILES
cat("--- Starting to organize and copy files ---\n")
for (i in 1:nrow(labels_df)) {
  file_id <- labels_df$id[i]
  infection_status <- labels_df$infection[i]
  
  # Construct the source file path (e.g., .../1/1.wav)
  source_file_path <- file.path(source_data_root, file_id, paste0(file_id, ".wav"))
  
  if (file.exists(source_file_path)) {
    # Define the destination folder based on infection status
    destination_folder <- ifelse(infection_status == 1, sick_dir_output, healthy_dir_output)
    
    # Create a new, unique filename to avoid conflicts (e.g., "1_1.wav")
    unique_filename <- paste0(gsub("/", "_", file_id), ".wav")
    destination_file_path <- file.path(destination_folder, unique_filename)
    
    # Copy the file
    file.copy(source_file_path, destination_file_path)
    cat(paste("Copied:", source_file_path, "->", destination_file_path, "\n"))
    
  } else {
    warning(paste("File not found:", source_file_path))
  }
}

cat("--- Data organization complete! ---\n")
cat(paste("Sick files are in:", sick_dir_output, "\n"))
cat(paste("Healthy files are in:", healthy_dir_output, "\n"))
cat("You can now run the main CNN script.\n")