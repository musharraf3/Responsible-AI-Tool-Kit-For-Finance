from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import matplotlib
import openai
from dotenv import load_dotenv
# Set OpenAI API key
openai.api_key = os.getenv("")
matplotlib.use('Agg')  # Use non-GUI backend

load_dotenv()  # Load environment variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API Key: {openai.api_key}")


openai.api_key = ""

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Route: Home Page
@app.route('/')
def home():
    return render_template('index.html')

def calculate_bias(features, target):
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric

    # Dynamic column naming based on the number of features
    column_names = [f'Feature_{i}' for i in range(features.shape[1])]

    # Create DataFrame from features and target
    df = pd.DataFrame(features, columns=column_names)
    df['Target'] = target  # Add target column to DataFrame

    # Define privileged and unprivileged groups
    privileged_group = [{'Feature_0': 1}]  # Replace with the index of the protected attribute
    unprivileged_group = [{'Feature_0': 0}]  # Replace with the index of the protected attribute

    # Create BinaryLabelDataset
    dataset = BinaryLabelDataset(
        df=df,
        label_names=['Target'],  # Name of the target column
        protected_attribute_names=['Feature_0'],  # Replace with the actual protected attribute
        favorable_label=1,
        unfavorable_label=0
    )

    # Compute fairness metrics
    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=privileged_group,
        unprivileged_groups=unprivileged_group
    )

    return {
        'Disparate Impact': metric.disparate_impact(),
        'Statistical Parity Difference': metric.statistical_parity_difference()
    }

def visualize_bias(original_bias, combined_bias, plots_dir):
    metrics = ['Disparate Impact', 'Statistical Parity Difference']
    original_values = [original_bias[metric] for metric in metrics]
    combined_values = [combined_bias[metric] for metric in metrics]

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(metrics))

    plt.bar(x, original_values, width=bar_width, label='Original Data', color='blue')
    plt.bar(x + bar_width, combined_values, width=bar_width, label='Combined Data', color='orange')
    plt.xticks(x + bar_width / 2, metrics)
    plt.ylabel('Metric Value')
    plt.title('Bias Metrics: Original vs Combined Data')
    plt.legend()

    plot_path = os.path.join(plots_dir, 'bias_comparison.png')
    plt.savefig(plot_path)
    plt.close()

# Route: Upload Dataset
@app.route('/upload', methods=['POST'])
def upload():
    global uploaded_file_name  # Use global variable to track file name
    if 'file' not in request.files:
        return "No file found"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    uploaded_file_name = file.filename  # Save the file name globally
    return f"File {file.filename} uploaded successfully!"

# Route: Data Preprocessing
@app.route('/preprocess', methods=['POST'])
def preprocess():
    global uploaded_file_name  # Access the uploaded file name
    if not uploaded_file_name:
        return "No file uploaded!"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file_name)

    # Load the dataset
    data = pd.read_csv(file_path)

    # Handle missing values for numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    # Handle missing values for non-numeric columns
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns
    data[non_numeric_cols] = data[non_numeric_cols].fillna(data[non_numeric_cols].mode().iloc[0])

    # Encode categorical columns (e.g., Gender, Married)
    if 'Gender' in data.columns:
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    if 'Married' in data.columns:
        data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})
    if 'Loan_Status' in data.columns:
        data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})

    # Save preprocessed data
    processed_path = os.path.join('data', 'preprocessed_data.csv')
    data.to_csv(processed_path, index=False)

    return jsonify({"message": "Data preprocessed successfully!", "processed_path": processed_path})


# Route: Generate Synthetic Data (GAN)
@app.route('/generate_data', methods=['POST'])
def generate_data():
    # Load the preprocessed dataset
    preprocessed_path = os.path.join('data', 'preprocessed_data.csv')
    if not os.path.exists(preprocessed_path):
        return render_template('index.html', message="Preprocessed data not found.")

    # Load data and ensure only numeric columns
    data = pd.read_csv(preprocessed_path)
    numeric_data = data.select_dtypes(include=['number'])
    X = numeric_data.values
    target_column = 'Loan_Status'  # Replace with your target column name
    target = data[target_column].values

    # Define GAN model
    input_dim = X.shape[1]
    generator = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dense(256, activation='relu'),
        Dense(input_dim, activation='tanh')
    ])
    generator.compile(optimizer='adam', loss='binary_crossentropy')

    # Generate synthetic data
    num_samples = X.shape[0]
    noise = np.random.normal(0, 1, (num_samples, input_dim))
    synthetic_data = generator.predict(noise)

    # Combine original and synthetic data
    combined_data = np.vstack([X, synthetic_data])
    combined_target = np.hstack([target, target])  # Duplicate target for synthetic data

    # Save synthetic data
    synthetic_path = os.path.join('data', 'synthetic_data.csv')
    pd.DataFrame(synthetic_data, columns=numeric_data.columns).to_csv(synthetic_path, index=False)

    # Compare distributions
    original_mean = np.mean(X, axis=0).tolist()
    synthetic_mean = np.mean(synthetic_data, axis=0).tolist()

    # Create visualizations for distributions
    plots_dir = os.path.join('static', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for i, column in enumerate(numeric_data.columns):
        plt.figure(figsize=(8, 6))
        plt.hist(X[:, i], bins=20, alpha=0.5, label='Original', color='blue')
        plt.hist(synthetic_data[:, i], bins=20, alpha=0.5, label='Synthetic', color='orange')
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.legend()
        plot_path = os.path.join(plots_dir, f"{column}_comparison.png")
        plt.savefig(plot_path)
        plt.close()

    # Calculate and visualize bias metrics
    original_bias = calculate_bias(X, target)
    combined_bias = calculate_bias(combined_data, combined_target)
    visualize_bias(original_bias, combined_bias, plots_dir)

    return render_template(
        'index.html',
        message="Synthetic data generated successfully! Visualizations created.",
        original_mean=f"Original Data Mean: {original_mean}",
        synthetic_mean=f"Synthetic Data Mean: {synthetic_mean}",
        plots=os.listdir(plots_dir)
    )

# Route: Train Predictive Model
@app.route('/train_model', methods=['POST'])
def train_model():
    # Load synthetic data and target variable
    synthetic_data_path = os.path.join('data', 'synthetic_data.csv')
    preprocessed_data_path = os.path.join('data', 'preprocessed_data.csv')

    if not os.path.exists(synthetic_data_path) or not os.path.exists(preprocessed_data_path):
        return render_template('index.html', message_1="Synthetic or preprocessed data not found.")

    X = pd.read_csv(synthetic_data_path).values
    y = pd.read_csv(preprocessed_data_path)['Loan_Status'].values

    # Ensure target variable has the same length as synthetic data
    if X.shape[0] != y.shape[0]:
        return render_template('index.html', message_1="Synthetic data and target variable sizes do not match.")

    # Train-test split
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Render template with results
    return render_template(
        'index.html',
        message_1="Model trained successfully!",
        accuracy=f"Accuracy: {accuracy * 100:.2f}%"
    )

@app.route('/explain_bias', methods=['POST'])
def explain_bias():
    try:
        data = request.json
        original_bias = data.get('original_bias')
        combined_bias = data.get('combined_bias')

        if not original_bias or not combined_bias:
            raise ValueError("Bias metrics are missing in the request.")

        # Prepare the prompt
        prompt = f"""
        I have calculated bias metrics for a dataset. The metrics are:
        Original Dataset:
        - Disparate Impact: {original_bias['Disparate Impact']}
        - Statistical Parity Difference: {original_bias['Statistical Parity Difference']}
        Combined Dataset:
        - Disparate Impact: {combined_bias['Disparate Impact']}
        - Statistical Parity Difference: {combined_bias['Statistical Parity Difference']}

        Explain these results in simple terms. What do they indicate about fairness in loan approvals? How has the synthetic data impacted fairness? Provide suggestions for improving fairness further.
        """

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains AI fairness metrics."},
                {"role": "user", "content": prompt}
            ]
        )
        explanation = response['choices'][0]['message']['content']
        print(f"Generated Explanation: {explanation}")  # Debugging

        return jsonify({"explanation": explanation})

    except Exception as e:
        print(f"Error in explain_bias: {e}")
        return jsonify({"error": str(e)})

@app.route('/get_bias_metrics', methods=['GET'])
def get_bias_metrics():
    try:
        # Assuming the bias metrics are already calculated during the data generation step
        preprocessed_path = os.path.join('data', 'preprocessed_data.csv')
        synthetic_data_path = os.path.join('data', 'synthetic_data.csv')

        if not os.path.exists(preprocessed_path) or not os.path.exists(synthetic_data_path):
            return jsonify({"error": "Preprocessed or synthetic data not found."})

        # Load original and synthetic data
        original_data = pd.read_csv(preprocessed_path).select_dtypes(include=['number']).values
        synthetic_data = pd.read_csv(synthetic_data_path).values

        # Assuming target is Loan_Status
        target = pd.read_csv(preprocessed_path)['Loan_Status'].values
        combined_data = np.vstack([original_data, synthetic_data])
        combined_target = np.hstack([target, target])

        # Calculate bias metrics
        original_bias = calculate_bias(original_data, target)
        combined_bias = calculate_bias(combined_data, combined_target)

        return jsonify({
            "original_bias": original_bias,
            "combined_bias": combined_bias
        })

    except Exception as e:
        print(f"Error in get_bias_metrics: {e}")
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
