# junction-event-camera-challenge

## Purpose
This project allows drone position and trajectory prediction with the usage of event camera data.

## Usage
To run the code, use the following command:

```bash
python <main.py_file_path> <dat_file_path>
```

## Requirements
You need to have the `evio` packages in your virtual environment's site packages to run the script. You can get it from [here](https://github.com/ahtihelminen/evio.git).
## Installation

To set up your environment, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone <repository_link>
    ```

2. **Navigate to the project directory**:
    ```bash
    cd <project_directory>
    ```

3. **Create a virtual environment** (if you haven't already):
    ```bash
    python -m venv venv
    ```

4. **Activate the virtual environment**:
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

5. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```
6. **Move evio package to site packages**
    Move the evio/src/evio folder to venv\Lib\site-packages folder

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome contributions that improve the functionality and performance of the code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.