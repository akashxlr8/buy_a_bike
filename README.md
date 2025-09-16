# Buy A Bike - Interactive Evaluation Studio

## Overview
The "Buy A Bike" app is an interactive evaluation studio designed to help users explore, filter, and compare motorcycles in the 350-650cc segment. It provides visual analytics, a flexible chart studio, a bike comparison matrix with meaningful delta coloring, and manufacturer-level insights.

## Features

- Overview dashboard with summary metrics and ranking visualizations.
- Interactive Bike Explorer with column guide and downloadable filtered dataset.
- Bike-to-Bike Comparison Matrix with meaningful delta coloring and baseline selection.
- Chart Studio for custom visualizations (scatter, bar, histogram, box, violin).
- Manufacturer insights and rankings.

## Data

### Source
The app uses the CSV file `advanced_motorcycle_evaluation_complete.csv` located in the repository root. This dataset was curated for the 350–650cc motorcycle segment and contains manufacturer, model, technical specs, pricing, and derived scores.

### Schema (columns)
Below are the most commonly used columns. If your dataset contains additional columns they will still be available in the explorer.

- `manufacturer` (string): Brand name, e.g., "Royal Enfield".
- `model` (string): Model name.
- `price` (numeric): Retail price in local currency.
- `engine_cc` (numeric): Engine displacement in cc.
- `power` (numeric): Peak power (PS or bhp, as provided).
- `torque` (numeric): Peak torque (Nm).
- `weight` (numeric): Kerb/wet weight in kg.
- `mileage` (numeric): Real-world/claimed mileage (kmpl or mpg standardized).
- `braking` (string): Braking system description, e.g., "ABS, single disc".
- `cooling` (string): Engine cooling type, e.g., "air-cooled".
- `overall_score` (numeric): Composite weighted score used for ranking.

If you want to extend the dataset, keep these types in mind. Numeric columns should be parseable by `pandas.to_numeric`.

## Installation

### Requirements
- Python 3.8+ recommended
- Install required packages. A `requirements.txt` is recommended but may be missing — use the list below if needed:

```bash
pip install streamlit pandas numpy plotly matplotlib
```

### Run the app

```bash
streamlit run app.py
```

Note: On Windows with Git Bash, use the same commands in a bash shell.

## Configuration & Customization

- To change default filters or default baseline bike, open `app.py` and look for the `DEFAULTS` or sidebar initialization block near the top.
- If you change column names in the CSV, update the `COLUMN_MAP` or any references to raw column names inside `app.py`.
- Add additional metrics: compute new numeric columns in the CSV or add on-the-fly derived columns using `pandas` in `app.py`.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d6248290-c978-4350-ad69-82759a582219" />

## Examples

1) Quick compare

 - Open the app, go to "Compare Bikes", select a baseline bike and 2-4 others. Observe the delta matrix colors where green indicates a positive change for metrics that are "higher is better" and red otherwise.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d87311e4-e873-4fa5-b322-7872d2c09043" />

2) Custom chart

 - Go to "Chart Studio", choose `price` on X, `power` on Y, color by `manufacturer`, and size by `power/weight` to visualize value vs performance.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/13f3edca-3a9e-4c5e-9506-5f4b54f554f4" />

## Troubleshooting

- App won't start: make sure your Python version is 3.8+ and required packages are installed.
- CSV parse errors: open `advanced_motorcycle_evaluation_complete.csv` in a spreadsheet app and verify numeric columns don't have stray characters (₹, commas). Remove or clean them and retry.
- Color/Styling not as expected: pandas Styler may render differently across Streamlit versions. Update Streamlit with `pip install --upgrade streamlit`.

## FAQ

- Q: Can I use a different dataset?
  - A: Yes. Place your CSV in the repo root and update the file name in `app.py`.

- Q: Are all columns required?
  - A: No. Missing optional columns will be ignored, but core columns like `manufacturer`, `model`, and relevant numeric columns are recommended.

## Development & Contributing

If you'd like to contribute:

1. Fork the repository and create a feature branch.
2. Add tests for new features where applicable.
3. Open a PR describing the change.

Coding conventions and recommendations:

- Use `pandas` for data transformations and keep UI logic in `app.py` minimal and well-commented.
- When adding visualization dependencies prefer `plotly` for interactive plots.

## License
This project is distributed under the MIT License. See `LICENSE` for full details.

## Credits

- Created and maintained by Akash.
- Data curated from multiple public sources and manufacturer specs; verify with official spec sheets before making purchasing decisions.

## Contact

Report issues via the repository Issues tab, or reach out to Akash via email: akash@example.com

