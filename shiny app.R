#
# Title: Poultry Vocalization Prediction Shiny App
#
# Description: This Shiny application loads the pre-trained 'torch' model
# and allows a user to upload a .wav file to get a prediction of whether
# the sound is from a 'Healthy' or 'Sick' chicken, or if it's 'None' (noise).
#
# Instructions:
# 1. Make sure you have run the main training script to generate the
#    'poultry_model_artifacts.rds' file.
# 2. Place this 'app.R' file in the SAME directory as the .rds file.
# 3. Install Shiny and the required packages: install.packages(c("shiny", "torch", "tuneR"))
# 4. Run the app by opening this script in RStudio and clicking "Run App",
#    or by using the command: shiny::runApp()
#

library(shiny)
library(torch)
library(tuneR)

#
# I. Load Model and Artifacts
#
# Load the saved model state and category labels.
# Define the model architecture again so we can load the state into it.
#

# Check if model artifacts file exists
if (!file.exists("poultry_model_artifacts.rds")) {
  stop("Model artifacts not found. Please run the training script first to generate 'poultry_model_artifacts.rds'.")
}

# Load the artifacts
artifacts <- torch_load("poultry_model_artifacts.rds")
CATEGORIES <- artifacts$categories
model_state_dict <- artifacts$model_state_dict

# Re-define the network architecture (must be identical to the training script)
Net <- nn_module(
  "Net",
  initialize = function() {
    self$conv1 <- nn_conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding = 1)
    self$conv2 <- nn_conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
    self$conv3 <- nn_conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
    self$fc1 <- nn_linear(in_features = 128 * 1 * 21, out_features = 128)
    self$fc2 <- nn_linear(in_features = 128, out_features = 3)
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

# Instantiate the model and load the trained weights
model <- Net()
model$load_state_dict(model_state_dict)
model$eval() # Set the model to evaluation mode

#
# II. Define Feature Extraction Function
#
# This function must be identical to the one used in the training script
# to ensure data is processed consistently.
#

N_MFCC <- 13
MAX_PAD_LEN <- 174

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


#
# III. Shiny UI (User Interface)
#
# Define the layout of the web application.
#

ui <- fluidPage(
  titlePanel("Poultry Vocalization Classifier"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("audio_file", "Upload a .wav Audio File",
                accept = c("audio/wav", "audio/x-wav")),
      
      actionButton("predict_button", "Classify Sound", class = "btn-primary")
    ),
    
    mainPanel(
      h3("Prediction Result"),
      # Use a div with a specific style for the output
      div(
        textOutput("prediction_output"),
        style = "font-size: 24px; font-weight: bold; color: #337ab7; margin-top: 20px;"
      )
    )
  )
)


#
# IV. Shiny Server (Application Logic)
#
# Define the backend logic that processes the uploaded file and makes a prediction.
#

server <- function(input, output) {
  
  # A reactive value to store the prediction result
  prediction_result <- eventReactive(input$predict_button, {
    
    # Ensure a file has been uploaded
    req(input$audio_file)
    
    # Get the path of the uploaded file
    file_path <- input$audio_file$datapath
    
    # Process the audio file to get features
    features <- extract_features(file_path)
    
    if (is.null(features)) {
      return("Error: Could not process the audio file.")
    }
    
    # Convert features to a torch tensor with the correct dimensions
    # (batch_size, channels, height, width)
    features_tensor <- torch_tensor(features, dtype = torch_float())$unsqueeze(1)$unsqueeze(1)
    
    # Make a prediction
    with_no_grad({
      output <- model(features_tensor)
      prediction_index <- torch_argmax(output, dim = 2)$item()
    })
    
    # Return the predicted class name
    return(CATEGORIES[prediction_index])
  })
  
  # Render the prediction result in the UI
  output$prediction_output <- renderText({
    prediction_result()
  })
}

# Run the application
shinyApp(ui = ui, server = server)
