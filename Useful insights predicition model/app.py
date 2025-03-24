from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.express as px
import plotly.utils
import json
import os
import pickle
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary folders if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['MODELS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Global variables to store data
current_data = None
trained_model = None
model_filename = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read and analyze the data
        current_data = pd.read_csv(filepath)
        
        # Generate basic statistics
        stats = {
            'rows': len(current_data),
            'columns': len(current_data.columns),
            'column_names': current_data.columns.tolist(),
            'dtypes': current_data.dtypes.astype(str).to_dict(),
            'missing_values': current_data.isnull().sum().to_dict(),
            'numeric_columns': current_data.select_dtypes(include=[np.number]).columns.tolist()
        }
        
        return jsonify(stats)
    
    return jsonify({'error': 'Invalid file format'})

@app.route('/preview', methods=['GET'])
def preview_data():
    if current_data is None:
        return jsonify({'error': 'No data uploaded'})
    
    preview = current_data.head(5).to_dict(orient='records')
    return jsonify(preview)

@app.route('/analyze', methods=['GET'])
def analyze_data():
    if current_data is None:
        return jsonify({'error': 'No data uploaded'})
    
    # Enhanced data analysis
    analysis = {
        'numerical_columns': current_data.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': current_data.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': current_data.isnull().sum().to_dict(),
        'summary_stats': current_data.describe().to_dict(),
        'correlation_matrix': current_data.select_dtypes(include=[np.number]).corr().to_dict(),
        'unique_values': {col: current_data[col].nunique() for col in current_data.columns}
    }
    
    return jsonify(analysis)

@app.route('/train', methods=['POST'])
def train_model():
    global trained_model, model_filename, current_data
    
    data = request.json
    target_column = data.get('target_column')
    model_type = data.get('model_type')
    
    if target_column not in current_data.columns:
        return jsonify({'error': 'Invalid target column'})
    
    # Prepare data
    X = current_data.drop(columns=[target_column])
    y = current_data[target_column]
    
    # Handle categorical variables
    X = pd.get_dummies(X)
    
    # Split data with 80-20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate split sizes
    split_info = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_percentage': 80,
        'test_percentage': 20
    }
    
    # Select and train model
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'random_forest_classifier':
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'random_forest_regressor':
        model = RandomForestRegressor(n_estimators=100)
    else:
        return jsonify({'error': 'Invalid model type'})
    
    model.fit(X_train, y_train)
    trained_model = model
    
    # Save the model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'model_{model_type}_{timestamp}.pkl'
    model_path = os.path.join(app.config['MODELS_FOLDER'], model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': X.columns.tolist(),
            'target_column': target_column,
            'model_type': model_type,
            'split_info': split_info
        }, f)
    
    # Calculate performance metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_score': train_score,
        'test_score': test_score,
        'split_info': split_info
    }
    
    # Add specific metrics based on model type
    if model_type in ['linear_regression', 'random_forest_regressor']:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        metrics.update({
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        })
    else:  # Classification metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        try:
            metrics.update({
                'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
                'test_precision': precision_score(y_test, y_test_pred, average='weighted'),
                'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
                'test_recall': recall_score(y_test, y_test_pred, average='weighted'),
                'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
                'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
            })
        except:
            # Fallback for non-binary/multiclass cases
            metrics.update({
                'train_accuracy': train_score,
                'test_accuracy': test_score
            })
    
    # Get feature importance
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X.columns, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        if len(model.coef_.shape) == 1:
            feature_importance = dict(zip(X.columns, abs(model.coef_)))
        else:
            # For multiclass, take average of absolute coefficients
            feature_importance = dict(zip(X.columns, abs(model.coef_).mean(axis=0)))
    
    return jsonify({
        'model_type': model_type,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'model_filename': model_filename
    })

@app.route('/download_model/<filename>')
def download_model(filename):
    if not filename or not os.path.exists(os.path.join(app.config['MODELS_FOLDER'], filename)):
        return jsonify({'error': 'Model not found'})
    
    return send_file(
        os.path.join(app.config['MODELS_FOLDER'], filename),
        as_attachment=True,
        download_name=filename
    )

@app.route('/visualize', methods=['GET'])
def visualize_data():
    if current_data is None:
        return jsonify({'error': 'No data uploaded'})
    
    insights = []
    
    # Numerical columns distribution
    for col in current_data.select_dtypes(include=[np.number]).columns:
        fig = px.histogram(current_data, x=col, title=f'Distribution of {col}')
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#ffffff'}
        )
        insights.append({
            'type': 'histogram',
            'title': f'Distribution of {col}',
            'plot': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
        })
    
    # Correlation matrix
    correlation = current_data.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(correlation, 
                    title='Correlation Matrix',
                    color_continuous_scale='RdBu')
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'}
    )
    insights.append({
        'type': 'heatmap',
        'title': 'Correlation Matrix',
        'plot': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    })
    
    return jsonify(insights)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
