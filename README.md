# AI_801_Final_Project (How to Run)

# The below instructions will walk a user through the process whether using the .py file or the Jupyter Notebook

# First, load all necessary libraries

# Adjust the parent directory  THese diretories are for the entire project.  This will need to be adjusted based on where the files are stored on your computer.

# The next block of code creates all necessary fised variables for use in the remiander of the project

# The next few steps are exploratory analytics.

    # Get the min and max number of images from the dataset
    # Test the sizes of each of the images in the training set 

# The next steps deal with identifying and removing duplicate images.  If you are using the dataset provided, this does not need to be run
    # Identify duplicate images in the dataset
    # Remove duplicate images

# The next steps deal with balancing the datasets after removing duplicates.  If you are using the dataset provided, this does not need to be run

    # Create the necessary folders where the augmented dataset will be stored
    # Select 120 images from each of the deduped folders
    # Check to ensure each of the new folders only have 120 images each

# The next few steps set up the function needed to get data loaded for analysis.

    # Create function to load pictures into numpy array
    # Function that loads the saves or loads the data in the appropriate format for analysis
    # Create the learning rate scheduler

# Model 1
    # Build the first CNN Model
    # Function to run the model
    # Run the model
    # Create variables for graphing
    # Plot the training and validation accuracy and loss
    # Create precision, recall, and F1 scores

# Keras Tuner
# NOTE: Keras tuner takes a lot of memory and a long time to run.  The optimal hpyperparameters have been saved in the keras_tuner_results folder for review
    # Build the Keras Tuner Model
    # Create a random search object and establish an early stopping parameter
    # Run Keras Tuner Model
    # Get the optimal hyperparameters

# Model 2 
    # Develop model based off Keras Tuner results
    # Run the tuned model
    # Create variables for graphing
    # Plot the training and validation accuracy and loss
    # Create precision, recall, and F1 scores

# Model 3
    # Create EfficientNet Model
    # Create Function to run EfficientNet Model
    # Run EfficientNet Model
    # Create variables for graphing
    # Plot the training and validation accuracy and loss
    # Create precision, recall, and F1 scores
    # Plot example results of first 16 species
