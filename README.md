# Unsupervised Clustering of Hotel Bookings

**Segmenting Business and Leisure Travellers Based on Booking Characteristics** *Unsupervised Learning · 2025/2026 · Group 3 — P2*

**Team:** 
* Rogério Soares (73803)
* Guilherme Simões (74154)
* Vasco Horta (73476)

## Project Overview
This project executes a complete, reproducible unsupervised clustering pipeline on a real-world hotel booking dataset. The goal is to determine if hotel bookings can be partitioned into stable, interpretable profiles using features known strictly at booking creation, and whether these segments map to traditional business vs. leisure travel patterns.

## Repository Setup & Execution

### 1. Dataset Requirement
To comply with repository hygiene standards, the raw dataset is **not** committed to this repository. Before running the notebook, you must obtain the dataset and place it in the root directory.
* **Filename:** `hotel_bookings_course_release_v1.csv`

### 2. Environment Setup
We recommend running this project inside an isolated virtual environment. 

**Mac/Linux:**
```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt