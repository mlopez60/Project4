# Project4 - The Plant Doctors
## Team Members: Matthew Lopez, Mitchell Fairgrieve, Austin Rothe & Steven Carrasquillo

### The Problem
- Every year more and more people are getting into plants and gardening as a hobby.
- However not everyone has a green thumb, has all the knowledge needed to maintain healthy plants nor know how to identify and treat diseases that could be affecting their plants.

### Our Solution
- Develop a website using a machine learning algorithm where users can submit photos and can be provided an identified illness and course for treatment with 90% accuracy.
- Starting with more common plants and illnesses.

### Dataset & Libaries
- We found on Kaggle a dataset with 87K images of 38 different plants and diseases
- https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

- We used a variety of libraries to achieve our goal:
-- Tensorflow: platform for machine learning
-- Keras_Tuner: utilized for our hyperparameter search to find the best model for our needs
-- PIL(Pillow): python library for image processing
-- Flask: used to display interactive disease prediction webpage

### Our Process
1. Import files into training & validation directories.
2. Define model building function (create_model).
3. Instantiate the tuner.
4. Run the hyperparameter tuner search.
5. Using best model build (tuner.get_best_models), begin training model with training data & validation data.

   ![image](https://github.com/mlopez60/Project4/assets/98186160/4d1b5ba7-45eb-4f3f-9fdc-84dfb620c0db)

6. Normalize pixel values (normalization layer)
7. Build and Train Model 
8. View Accuracy vs Loss (accuracy: 0.9710)
    
   ![image](https://github.com/mlopez60/Project4/assets/98186160/84059329-b45e-49b2-8404-50b447f9d30e)
   
9. Use Model to make Plant Disease Predictions!!!


### Results

**Upload image of completed webpage**

