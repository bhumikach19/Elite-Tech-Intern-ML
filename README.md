# Elite - Movie Recommendation System

A sophisticated movie recommendation system that uses neural network-based collaborative filtering to provide personalized movie recommendations.

## Project Structure

```
Elite/
├── TASK 4/
│   ├── app.py              # Streamlit web application
│   ├── requirements.txt    # Project dependencies
│   └── TASK_4.ipynb        # Jupyter notebook with model development
```

## Features

- **Neural Network-based Collaborative Filtering**
  - User and movie embeddings
  - Customizable embedding dimensions
  - Real-time model training
  - Performance evaluation metrics

- **Interactive Web Interface**
  - User-friendly Streamlit dashboard
  - Real-time recommendations
  - Data visualization
  - Model performance analysis

- **Data Analysis**
  - Rating distribution visualization
  - User activity analysis
  - Model evaluation metrics
  - Training history visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Elite.git
cd Elite
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run TASK 4/app.py
```

2. Access the web interface:
   - Open your web browser
   - Navigate to `http://localhost:8501`

3. Using the application:
   - Upload your dataset (optional)
   - Configure model parameters
   - Train the model
   - Generate recommendations
   - View performance metrics

## Data Format

The application accepts CSV files with the following columns:
- `userId`: Unique identifier for users
- `movieId`: Unique identifier for movies
- `rating`: User's rating for the movie (numeric)

## Model Configuration

You can customize the following parameters:
- Embedding size
- Number of training epochs
- Batch size
- Number of recommendations

## Evaluation Metrics

The system provides:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Training history visualization
- Actual vs predicted ratings comparison

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow for the deep learning framework
- Streamlit for the web interface
- Plotly for interactive visualizations
- Scikit-learn for evaluation metrics

## Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/Elite](https://github.com/yourusername/Elite) 