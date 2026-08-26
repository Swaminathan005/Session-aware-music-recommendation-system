# Session-aware Music Recommendation System

A music recommendation system that incorporates session-aware contextual information to provide personalized music recommendations.

## Overview

This project focuses on building a music recommendation system that considers:
- User listening sessions
- Temporal context
- User preferences
- Music characteristics

## Project Structure

- `archive.zip` - Compressed data archive
- `initial_data.csv` - Initial dataset for the recommendation system
- `user_char.csv` - User characteristics and metadata
- `try.ipynb` - Jupyter notebook for experimentation and analysis
- `archive/` - Extracted archive folder
- `.ipynb_checkpoints/` - Jupyter notebook checkpoints

## Data

The system uses two main datasets:
1. **Initial Data** - Base dataset for recommendations
2. **User Characteristics** - Metadata about users for personalization

## Setup

1. Clone the repository:
   ```bash
   git clone git@github.com:Swaminathan005/Session-aware-music-recommendation-system.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Session-aware-music-recommendation-system
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Open the Jupyter notebook to explore the data and run experiments:
```bash
jupyter notebook try.ipynb
```

## Features

- Session-aware recommendations
- User preference modeling
- Contextual filtering
- Collaborative filtering approaches

## Requirements

- Python 3.8+
- Jupyter Notebook
- Pandas, NumPy, Scikit-learn
- Additional packages listed in `requirements.txt`

## License

MIT License - see LICENSE file for details