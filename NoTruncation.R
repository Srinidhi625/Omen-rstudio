#
# Title: Poultry Vocalization CRNN with Attention in R using torch
#
# Description: This R script trains a CRNN with Attention using two separate datasets.
# It now uses DYNAMIC PADDING to utilize the full length of training audio files
# without truncation. During testing, it splits longer audio files into chunks
# for a more robust classification. The model architecture adapts automatically.
# This script has NO PYTHON DEPENDENCY.
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


#
# IV. Feature Extraction and Dynamic Padding
#
# This section processes audio files without truncation. It finds the maximum
# length in the training set and pads all training files to that length.
#

# Set the number of MFCCs to extract.
N_MFCC <- 13

# Function to extract full, unpadded features from a file
extract_full_features <- function(file_path) {
  tryCatch({
    wave <- tuneR::readWave(file_path)
    if (wave@stereo) {
      wave <- tuneR::mono(wave, "left")
    }
    mfccs <- t(tuneR::melfcc(wave, numcep = N_MFCC, sr = wave@samp.rate))
    return(mfccs)
  }, error = function(e) {
    print(paste("Error processing", file_path, ":", e$message))
    return(NULL)
  })
}

# --- Training Data Processing ---
print("Extracting features from training data...")
train_features_list <- lapply(train_df$file_path, extract_full_features)

# Remove any NULLs that resulted from errors
valid_train_indices <- !sapply(train_features_list, is.null)
train_features_list <- train_features_list[valid_train_indices]
train_df <- train_df[valid_train_indices, ]

# Find the maximum length (number of time steps) in the training set
max_len <- max(sapply(train_features_list, ncol))
print(paste("Maximum feature length in training set:", max_len))

# Function to pad features to the max length
pad_features <- function(mfccs, max_pad_len) {
  pad_width <- max_pad_len - ncol(mfccs)
  if (pad_width > 0) {
    mfccs <- cbind(mfccs, matrix(0, nrow = nrow(mfccs), ncol = pad_width))
  }
  # Note: No truncation is performed now
  return(mfccs)
}

# Apply padding to all training features
print("Padding training features...")
padded_train_features <- lapply(train_features_list, pad_features, max_pad_len = max_len)

# Convert the padded training features into a 3D array
train_features_array <- array(unlist(padded_train_features), dim = c(N_MFCC, max_len, length(padded_train_features)))
train_features_array <- aperm(train_features_array, c(3, 1, 2))


#
# V. Data Preparation for torch
#
# We create a dataset and dataloader for the training data only.
# Test data will be processed individually.
#

# Custom dataset definition for training
training_dataset_torch <- dataset(
  name = "training_dataset_torch",
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

train_dataset <- training_dataset_torch(train_features_array, train_df$label)

# Create dataloader for training
BATCH_SIZE <- 32
train_dataloader <- dataloader(train_dataset, batch_size = BATCH_SIZE, shuffle = TRUE)
print(paste("Training samples:", length(train_dataset)))


#
# VI. Model Building with torch (CRNN + Attention Architecture)
#
# The model is now initialized dynamically based on the calculated max_len.
#

CRNN_Attention_Net <- nn_module(
  "CRNN_Attention_Net",
  initialize = function(n_mfcc_val, max_pad_len_val) {
    self$conv1 <- nn_conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
    self$conv2 <- nn_conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
    self$conv3 <- nn_conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
    
    # Dynamically calculate dimensions after pooling
    final_height <- floor(n_mfcc_val / 8)
    
    self$lstm <- nn_lstm(input_size = 128 * final_height, hidden_size = 64, batch_first = TRUE, bidirectional = TRUE)
    self$attention_weights <- nn_linear(in_features = 128, out_features = 1)
    self$tanh <- nn_tanh()
    self$fc1 <- nn_linear(in_features = 128, out_features = 64)
    self$fc2 <- nn_linear(in_features = 64, out_features = 3)
  },
  forward = function(x) {
    x <- self$conv1(x) %>% nnf_relu() %>% nnf_max_pool2d(2)
    x <- self$conv2(x) %>% nnf_relu() %>% nnf_max_pool2d(2)
    x <- self$conv3(x) %>% nnf_relu() %>% nnf_max_pool2d(2)
    x <- x$permute(c(1, 4, 2, 3))
    x <- torch_flatten(x, start_dim = 3)
    lstm_out <- self$lstm(x)[[1]]
    attention_scores <- self$attention_weights(lstm_out) %>% self$tanh()
    attention_weights <- nnf_softmax(attention_scores, dim = 2)
    context_vector <- torch_sum(lstm_out * attention_weights, dim = 2)
    x <- context_vector %>%
      self$fc1() %>% nnf_relu() %>% nnf_dropout(p = 0.5) %>% self$fc2()
    return(x)
  }
)

# Instantiate the model, passing the dynamic feature parameters
model <- CRNN_Attention_Net(n_mfcc_val = N_MFCC, max_pad_len_val = max_len)

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
  
  avg_train_loss <- train_loss / length(train_dataloader)
  cat(sprintf("Epoch %d/%d, Train Loss: %.4f\n", epoch, EPOCHS, avg_train_loss))
}

#
# VIII. Detailed Model Performance Analysis with Chunking
#
# This section now processes each test file by splitting it into chunks.
#

print("Calculating detailed performance metrics on the test set using chunking...")

model$eval()
all_preds <- c()
all_targets <- c()

with_no_grad({
  for (i in 1:nrow(test_df)) {
    file_path <- test_df$file_path[i]
    true_label <- as.integer(test_df$label[i])
    
    # Extract full features for the test file
    full_mfccs <- extract_full_features(file_path)
    
    if (is.null(full_mfccs)) next
    
    # Split the MFCCs into chunks of the size the model was trained on
    num_chunks <- ceiling(ncol(full_mfccs) / max_len)
    chunk_preds <- c()
    
    for (j in 1:num_chunks) {
      start_col <- (j - 1) * max_len + 1
      end_col <- min(j * max_len, ncol(full_mfccs))
      
      chunk <- full_mfccs[, start_col:end_col, drop = FALSE]
      
      # Pad the last chunk if it's smaller
      chunk <- pad_features(chunk, max_len)
      
      # Convert to tensor and add batch/channel dimensions
      chunk_tensor <- torch_tensor(chunk, dtype = torch_float())$unsqueeze(1)$unsqueeze(1)
      
      # Get prediction for the chunk
      output <- model(chunk_tensor)
      pred <- torch_argmax(output, dim = 2)$item()
      chunk_preds <- c(chunk_preds, pred)
    }
    
    # Aggregate predictions for the file using a majority vote
    final_prediction <- as.integer(names(which.max(table(chunk_preds))))
    
    all_preds <- c(all_preds, final_prediction)
    all_targets <- c(all_targets, true_label)
  }
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
#
print("Saving model and artifacts...")
model$eval() 
artifacts_to_save <- list(
  model_state_dict = model$state_dict(),
  categories = CATEGORIES,
  n_mfcc = N_MFCC,
  # Save the max_len used for training, as this is now the required chunk size
  max_pad_len = max_len, 
  model_type = "CRNN_Attention"
)
torch_save(artifacts_to_save, "poultry_model_artifacts_crnn_attention.rds")
print("Model artifacts saved to poultry_model_artifacts_crnn_attention.rds")

