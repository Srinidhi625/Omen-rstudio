#
# Title: Poultry Vocalization CRNN Classification in R using torch
#
# Description: This R script trains a Convolutional Recurrent Neural Network (CRNN)
# using two separate datasets (training and testing). This hybrid model uses CNN
# layers for feature extraction and an LSTM layer to model temporal sequences,
# as suggested by recent research. The script is written in pure R with torch and
# has NO PYTHON DEPENDENCY.
#
# Dataset: https://data.mendeley.com/datasets/dy6gtvt4mk/2
#

#
# I. Setup and Installation
#
# In this section, we install the necessary R packages.
#

# --- One-time Setup ---
# 1. Install required R packages
# install.packages(c("torch", "tuneR", "caret", "e1071"))

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
# Please provide the full paths to your separate training and testing folders.
#

# --- USER INPUT REQUIRED ---
# Path to the folder containing the TRAINING data ('Healthy', 'Sick', 'None' subfolders)
TRAIN_DATA_PATH <- 'C:/Users/srini/OneDrive/Desktop/R/Omen-rstudio/SmartEars A Practical Framework for Poultry Respiratory Monitoring via Spectrogram-Based Audio Classification and AI-Assisted Labeling'
# Path to the folder containing the TESTING data ('Healthy', 'Sick', 'None' subfolders)
TEST_DATA_PATH <- 'C:/Users/srini/OneDrive/Desktop/R/Omen-rstudio/Chicken_Audio_Dataset' # <-- IMPORTANT: CHANGE THIS PATH
# -------------------------

# Check if the data directories exist
if (!dir.exists(TRAIN_DATA_PATH)) {
  stop(paste("Training data directory not found at:", TRAIN_DATA_PATH, ". Please ensure the path is correct."))
} else {
  print("Training data directory found.")
}

if (!dir.exists(TEST_DATA_PATH)) {
  stop(paste("Testing data directory not found at:", TEST_DATA_PATH, ". Please ensure the path is correct."))
} else {
  print("Testing data directory found.")
}


#
# III. Data Loading
#
# In this section, we'll load the files from their respective directories.
#

# Define the categories
CATEGORIES <- c("Healthy", "Sick", "None")

# Load training files
print("Loading training files...")
train_file_list <- list.files(TRAIN_DATA_PATH, recursive = TRUE, full.names = TRUE, pattern = "\\.wav$")
train_labels <- basename(dirname(train_file_list))
train_df <- data.frame(
  file_path = train_file_list,
  label = as.factor(train_labels),
  stringsAsFactors = FALSE
)
print(paste("Loaded", nrow(train_df), "training audio files."))

# Load testing files
print("Loading testing files...")
test_file_list <- list.files(TEST_DATA_PATH, recursive = TRUE, full.names = TRUE, pattern = "\\.wav$")
test_labels <- basename(dirname(test_file_list))
test_df <- data.frame(
  file_path = test_file_list,
  label = as.factor(test_labels),
  stringsAsFactors = FALSE
)
print(paste("Loaded", nrow(test_df), "testing audio files."))

# Combine into a single data frame for feature extraction
combined_df <- rbind(train_df, test_df)


#
# IV. Feature Extraction (MFCC)
#
# This section runs on the combined dataset.
#

# Set parameters for feature extraction
N_MFCC <- 30
MAX_PAD_LEN <- 300
extract_features <- function(file_path) {
  tryCatch({
    wave <- tuneR::readWave(file_path)
    if (wave@stereo) {
      wave <- tuneR::mono(wave, "left")
    }
    mfccs <- t(tuneR::melfcc(wave, numcep = N_MFCC, sr = wave@samp.rate))
    pad_width <- MAX_PAD_LEN - ncol(mfccs)
    if (pad_width > 0) {
      mfccs <- cbind(mfccs, matrix(0, nrow = nrow(mfccs), ncol = pad_width))
    } else {
      mfccs <- mfccs[, 1:MAX_PAD_LEN]
    }
    return(mfccs)
  }, error = function(e) {
    print(paste("Error processing", file_path, ":", e$message))
    return(NULL)
  })
}

# Apply the feature extraction to all files
print("Extracting features from the dataset...")
features_list <- lapply(combined_df$file_path, extract_features)

# Remove any NULLs that resulted from errors
valid_indices <- !sapply(features_list, is.null)
features_list <- features_list[valid_indices]
combined_df <- combined_df[valid_indices, ]

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
    self$data <- torch_tensor(features, dtype = torch_float())$unsqueeze(2)
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
full_dataset <- poultry_dataset(features_array, combined_df$label)

# Split the full dataset back into training and testing sets based on the original file counts
train_indices <- 1:nrow(train_df)
test_indices <- (nrow(train_df) + 1):nrow(combined_df)

train_dataset <- dataset_subset(full_dataset, train_indices)
test_dataset <- dataset_subset(full_dataset, test_indices)

# Create dataloaders
BATCH_SIZE <- 32
train_dataloader <- dataloader(train_dataset, batch_size = BATCH_SIZE, shuffle = TRUE)
test_dataloader <- dataloader(test_dataset, batch_size = BATCH_SIZE, shuffle = FALSE)

print(paste("Training samples:", length(train_dataset)))
print(paste("Testing samples:", length(test_dataset)))

#
# VI. Model Building with torch (CRNN Architecture)
#
# Define the CRNN model architecture using torch's nn_module.
#

CRNN_Net <- nn_module(
  "CRNN_Net",
  initialize = function() {
    # Convolutional layers for feature extraction
    self$conv1 <- nn_conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
    self$conv2 <- nn_conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
    self$conv3 <- nn_conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
    
    # Recurrent layer (LSTM) for sequence modeling
    # The input size will be the number of feature maps from the CNN (128) times the height (5)
    self$lstm <- nn_lstm(input_size = 128 * 5, hidden_size = 64, batch_first = TRUE)
    
    # Fully connected layers for classification
    self$fc1 <- nn_linear(in_features = 64, out_features = 32)
    self$fc2 <- nn_linear(in_features = 32, out_features = 3)
  },
  forward = function(x) {
    # Pass through convolutional layers
    x <- self$conv1(x) %>%
      nnf_relu() %>%
      nnf_max_pool2d(2)
    
    x <- self$conv2(x) %>%
      nnf_relu() %>%
      nnf_max_pool2d(2)
    
    x <- self$conv3(x) %>%
      nnf_relu() %>%
      nnf_max_pool2d(2)
    
    # Reshape for LSTM: (batch_size, channels, height, width) -> (batch_size, width, channels * height)
    # This treats the width (time dimension) as the sequence length.
    x <- x$permute(c(1, 4, 2, 3)) # (batch, width, channels, height)
    x <- torch_flatten(x, start_dim = 3) # (batch, width, channels * height)
    
    # Pass through LSTM
    # We only need the output of the last time step for classification
    lstm_out <- self$lstm(x)
    lstm_out <- lstm_out[[1]][, -1, ..] # Get the last time step's output
    
    # Pass through fully connected layers
    x <- lstm_out %>%
      self$fc1() %>%
      nnf_relu() %>%
      nnf_dropout(p = 0.5) %>%
      self$fc2()
    
    return(x)
  }
)

# Instantiate the model
model <- CRNN_Net()

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
# This section calculates and displays a confusion matrix and other
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
# IX. Save the Model
#
# This section saves the trained model and the category labels to a file.
# This file will be loaded by the Shiny app for making predictions.
#
print("Saving model and artifacts...")
model$eval() 
artifacts_to_save <- list(
  model_state_dict = model$state_dict(),
  categories = CATEGORIES,
  # Also save the parameters used, so the Shiny app can build the model correctly
  n_mfcc = N_MFCC,
  max_pad_len = MAX_PAD_LEN,
  model_type = "CRNN" # Add a model type identifier
)
torch_save(artifacts_to_save, "poultry_model_artifacts_crnn.rds")
print("Model artifacts saved to poultry_model_artifacts_crnn.rds")

