#
# Title: Poultry Vocalization Classification in R using torch (Python-Free)
#
# Description: This R script uses a pre-downloaded and extracted "SmartEars" dataset.
# It preprocesses the audio data by extracting MFCCs, trains a Convolutional
# Neural Network (CNN) using the 'torch' package, and saves the trained model
# for later use in a Shiny application. This script has NO PYTHON DEPENDENCY.
#
# Dataset: https://data.mendeley.com/datasets/dy6gtvt4mk/2
#

#
# I. Setup and Installation
#
# In this section, we install the necessary R packages. The 'torch' package
# requires a one-time installation of its core libraries. We will use 'tuneR'
# for all audio processing and 'caret' for detailed model evaluation.
#

# --- One-time Setup ---
# 1. Install required R packages
# install.packages(c("torch", "tuneR", "caret", "e1071")) # e1071 is needed by caret

# 2. Install the torch backend libraries (this is a one-time step)
# library(torch)
# install_torch()
# --------------------

# Load the required libraries for the session
library(torch)
library(tuneR)
library(caret)

#
# II. Verify Data Location
#
# This section assumes the dataset has already been extracted into the
# 'poultry_dataset/SmartEars_Dataset_V2' directory.
#

DATA_PATH <- file.path( "C:/Users/srini/OneDrive/Poultry/SmartEars A Practical Framework for Poultry Respiratory Monitoring via Spectrogram-Based Audio Classification and AI-Assisted Labeling"
)

# Check if the data directory exists
if (!dir.exists(DATA_PATH)) {
  stop(paste("Dataset directory not found at:", DATA_PATH, ". Please ensure the data is extracted correctly in the working directory."))
} else {
  print("Dataset directory found. Proceeding with data loading.")
}


#
# III. Data Loading
#
# In this section, we'll locate all the .wav files and create a data frame
# that maps each file to its correct label ('Healthy', 'Sick', 'None').
#

# Define the categories
CATEGORIES <- c("Healthy", "Sick", "None")

# Create a list of all .wav files and their corresponding labels
file_list <- list.files(DATA_PATH, recursive = TRUE, full.names = TRUE, pattern = "\\.wav$")
labels <- basename(dirname(file_list))

# Create a data frame to hold file paths and labels
data_df <- data.frame(
  file_path = file_list,
  label = as.factor(labels),
  stringsAsFactors = FALSE
)

print(paste("Loaded", nrow(data_df), "audio files."))


#
# IV. Feature Extraction (MFCC)
#
# This part is now corrected to use the 'tuneR' package directly, which
# provides robust handling of both mono and stereo audio files.
#

# Set parameters for feature extraction
N_MFCC <- 13
MAX_PAD_LEN <- 174

extract_features <- function(file_path) {
  tryCatch({
    # Read the wave file using tuneR
    wave <- tuneR::readWave(file_path)
    
    # If the file is stereo, convert it to mono by taking the left channel
    if (wave@stereo) {
      wave <- tuneR::mono(wave, "left")
    }
    
    # Calculate MFCCs using tuneR::melfcc.
    # The output of melfcc is already in the format (time x coeffs),
    # so we need to transpose it to get (coeffs x time).
    mfccs <- t(tuneR::melfcc(wave, numcep = N_MFCC, sr = wave@samp.rate))
    
    # Pad or truncate the MFCC matrix
    pad_width <- MAX_PAD_LEN - ncol(mfccs)
    if (pad_width > 0) {
      # Pad with zeros
      mfccs <- cbind(mfccs, matrix(0, nrow = nrow(mfccs), ncol = pad_width))
    } else {
      # Truncate
      mfccs <- mfccs[, 1:MAX_PAD_LEN]
    }
    return(mfccs)
  }, error = function(e) {
    print(paste("Error processing", file_path, ":", e$message))
    return(NULL)
  })
}

# Apply the feature extraction to all files
print("Extracting features from audio files...")
features_list <- lapply(data_df$file_path, extract_features)

# Remove any NULLs that resulted from errors
valid_indices <- !sapply(features_list, is.null)
features_list <- features_list[valid_indices]
data_df <- data_df[valid_indices, ]

# Convert the list of matrices into a 3D array and then to a torch tensor
features_array <- array(unlist(features_list), dim = c(N_MFCC, MAX_PAD_LEN, length(features_list)))
features_array <- aperm(features_array, c(3, 1, 2)) # (samples, coeffs, time_steps)

#
# V. Data Preparation for torch
#
# We create a custom dataset and dataloaders for use with torch.
#

# Custom dataset definition
poultry_dataset <- dataset(
  name = "poultry_dataset",
  
  initialize = function(features, labels) {
    # Add a channel dimension and convert to tensor
    self$data <- torch_tensor(features, dtype = torch_float())$unsqueeze(2)
    # Convert labels to 1-based integers. The torch R package
    # expects 1-based indexing for target labels.
    self$targets <- torch_tensor(as.integer(labels), dtype = torch_long())
  },
  
  .getitem = function(index) {
    list(x = self$data[index, ..], y = self$targets[index])
  },
  
  .length = function() {
    self$targets$size()[[1]]
  }
)

# Instantiate the dataset
full_dataset <- poultry_dataset(features_array, data_df$label)

# Split into training and testing sets
set.seed(42)
train_indices <- sample(1:length(full_dataset), size = floor(0.8 * length(full_dataset)))
test_indices <- setdiff(1:length(full_dataset), train_indices)

train_dataset <- dataset_subset(full_dataset, train_indices)
test_dataset <- dataset_subset(full_dataset, test_indices)

# Create dataloaders
BATCH_SIZE <- 32
train_dataloader <- dataloader(train_dataset, batch_size = BATCH_SIZE, shuffle = TRUE)
test_dataloader <- dataloader(test_dataset, batch_size = BATCH_SIZE, shuffle = FALSE)

print(paste("Training samples:", length(train_dataset)))
print(paste("Testing samples:", length(test_dataset)))

#
# VI. Model Building with torch
#
# Define the CNN model architecture using torch's nn_module.
#

Net <- nn_module(
  "Net",
  initialize = function() {
    # Using padding to better control dimensions through the network
    self$conv1 <- nn_conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
    self$conv2 <- nn_conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
    self$conv3 <- nn_conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
    
    # With padding=1 and kernel=3, dimensions are preserved before pooling.
    # Input: 13x174 -> Pool: 6x87
    # After conv2 -> Pool: 3x43
    # After conv3 -> Pool: 1x21
    # Flattened size: 128 * 1 * 21 = 2688
    self$fc1 <- nn_linear(in_features = 128 * 1 * 21, out_features = 128)
    self$fc2 <- nn_linear(in_features = 128, out_features = 3) # 3 classes
  },
  forward = function(x) {
    x <- self$conv1(x) %>%
      nnf_relu() %>%
      nnf_max_pool2d(2) %>%
      nnf_dropout(p = 0.25)
    
    x <- self$conv2(x) %>%
      nnf_relu() %>%
      nnf_max_pool2d(2) %>%
      nnf_dropout(p = 0.25)
    
    x <- self$conv3(x) %>%
      nnf_relu() %>%
      nnf_max_pool2d(2) %>%
      nnf_dropout(p = 0.25)
    
    x %>%
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      nnf_dropout(p = 0.5) %>%
      self$fc2()
  }
)

model <- Net()

#
# VII. Model Training with torch
#
# We now write a manual training loop.
#

EPOCHS <- 50
optimizer <- optim_adam(model$parameters)
loss_fn <- nn_cross_entropy_loss()

print("Starting model training...")
for (epoch in 1:EPOCHS) {
  model$train()
  train_loss <- 0
  
  coro::loop(for (batch in train_dataloader) {
    optimizer$zero_grad()
    output <- model(batch$x)
    loss <- loss_fn(output, batch$y)
    loss$backward()
    optimizer$step()
    train_loss <- train_loss + loss$item()
  })
  
  # Calculate average loss for the epoch
  avg_train_loss <- train_loss / length(train_dataloader)
  
  # Validation phase
  model$eval()
  test_loss <- 0
  correct <- 0
  
  with_no_grad({
    coro::loop(for (batch in test_dataloader) {
      output <- model(batch$x)
      test_loss <- test_loss + loss_fn(output, batch$y)$item()
      pred <- torch_argmax(output, dim = 2)
      correct <- correct + torch_sum(pred == batch$y)$item()
    })
  })
  
  avg_test_loss <- test_loss / length(test_dataloader)
  accuracy <- correct / length(test_dataset)
  
  cat(sprintf("Epoch %d/%d, Train Loss: %.4f, Val Loss: %.4f, Val Accuracy: %.4f\n",
              epoch, EPOCHS, avg_train_loss, avg_test_loss, accuracy))
}

#
# VIII. Detailed Model Performance Analysis
#
# This new section calculates and displays a confusion matrix and other
# detailed classification metrics for the test set.
#

print("Calculating detailed performance metrics on the test set...")

model$eval()
all_preds <- c()
all_targets <- c()

with_no_grad({
  coro::loop(for (batch in test_dataloader) {
    output <- model(batch$x)
    preds <- torch_argmax(output, dim = 2)
    
    # Append predictions and targets for this batch to the main vectors
    all_preds <- c(all_preds, as.numeric(preds))
    all_targets <- c(all_targets, as.numeric(batch$y))
  })
})

# Convert numeric predictions and targets back to factor levels for caret
predicted_labels <- factor(CATEGORIES[all_preds], levels = CATEGORIES)
actual_labels <- factor(CATEGORIES[all_targets], levels = CATEGORIES)

# Generate and print the confusion matrix and associated statistics
confusion_matrix_results <- caret::confusionMatrix(
  data = predicted_labels,
  reference = actual_labels
)

print(confusion_matrix_results)


#
# IX. Making Predictions
#
# Use the trained model to make a prediction on a single, random sample.
#

print("Making a prediction on a random sample...")
model$eval()
with_no_grad({
  # Get a random sample from the test set
  sample_index <- sample(1:length(test_dataset), 1)
  sample <- test_dataset[sample_index]
  
  # sample$x needs a batch dimension
  output <- model(sample$x$unsqueeze(1))
  
  prediction <- torch_argmax(output, dim = 2)$item()
  
  predicted_class <- CATEGORIES[prediction] # The prediction is already 1-based
  actual_class <- CATEGORIES[sample$y$item()]
  
  print(paste("Predicted class:", predicted_class))
  print(paste("Actual class:", actual_class))
})

#
# X. Save the Model
#
# This section saves the trained model and the category labels to a file.
# This file will be loaded by the Shiny app for making predictions.
#
print("Saving model and artifacts...")
# Ensure the model is in evaluation mode before saving
model$eval() 
# Create a list containing the model's state dictionary and the category labels
artifacts_to_save <- list(
  model_state_dict = model$state_dict(),
  categories = CATEGORIES
)
torch_save(artifacts_to_save, "poultry_model_artifacts.rds")
print("Model artifacts saved to poultry_model_artifacts.rds")


