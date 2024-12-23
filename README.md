# Classifier Comparison Project

This project compares the performance of six different classification algorithms using the Iris dataset. The goal is to demonstrate how various machine learning models perform on a standard dataset and analyze their results based on several performance metrics.

## Algorithms Compared

1. Decision Tree
2. Random Forest
3. Naive Bayes
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)
6. Gradient Boosting (newly added algorithm)

## Steps to Run the Code

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/ClassifierComparison.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ClassifierComparison
   ```

3. Make sure Python is installed on your system. Install the required libraries using pip:

   ```bash
   pip install pandas scikit-learn
   ```

4. Run the Python script to train and test the classifiers:

   ```bash
   python classifier_comparison.py
   ```

5. After running the script, a CSV file named `classifier_comparison_results.csv` will be created in the project directory. This file contains the performance metrics for all classifiers.

## Performance Metrics

The comparison is based on the following metrics:

- **Accuracy**: The proportion of correctly predicted samples.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positives identified among all actual positive samples.
- **F1 Score**: The harmonic mean of precision and recall, balancing both metrics.

## Results

The results of the comparison are printed in the console and saved in the `classifier_comparison_results.csv` file. You can analyze the data to see which algorithm performed best for this dataset.

## Notes

- This project uses the Iris dataset as an example. You can replace it with any other dataset by modifying the data loading section in the script.
- Gradient Boosting is included as the new algorithm to test against the five algorithms commonly discussed in the video.

## Feedback

Feel free to submit an issue or reach out if you have any questions or suggestions for improvement. Thank you for reviewing my project!