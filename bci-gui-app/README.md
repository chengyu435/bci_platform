# bci-gui-app

## Overview
The BCI GUI App is a Python-based graphical user interface application designed for analyzing EEG data. It allows users to select data files, choose specific channels, apply filters, and visualize brain connectivity through various plots.

## Features
- **File Selection**: Users can easily select EEG data files from their system.
- **Channel Selection**: Users can choose one or multiple EEG channels for analysis.
- **Filtering Options**: The application provides options to apply different filters to the selected data.
- **Connectivity Visualization**: Users can set thresholds for visualizing brain connectivity and view the results in graphical form.
- **Results Display**: The application displays both textual and graphical results of the analysis.

## Project Structure
```
bci-gui-app
├── src
│   ├── main.py            # Entry point of the application
│   ├── app.py             # Main application logic
│   ├── gui                # GUI components
│   │   ├── __init__.py    # Marks gui as a package
│   │   └── main_window.py  # Implementation of the main window
│   ├── services           # Services for data analysis
│   │   ├── __init__.py    # Marks services as a package
│   │   └── analysis_service.py # Logic for data analysis
│   └── utils              # Utility functions
│       └── __init__.py    # Marks utils as a package
├── data                   # Directory for EEG data files
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── .gitignore             # Files to ignore by Git
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd bci-gui-app
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```
   python src/main.py
   ```
2. Follow the on-screen instructions to select data files, choose channels, apply filters, and visualize results.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.