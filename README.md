## Amino Acid Repeats: Analysis & Detection

This repository contains the presentation slides and Python scripts related to the analysis and detection of Amino Acid Repeats (AARs) in protein sequences. The project explores various computational methodologies for identifying these repeats and discusses their biological significance and implications.

---

### Features & Methodologies

The Python scripts in this repository implement different strategies for detecting amino acid repeats, as discussed in the accompanying presentation:

* **Self-Comparison Alignment**: Utilizes sequence alignment techniques to identify direct and inverted repeats within a protein sequence.
* **Fourier Transform Pattern Recognition**: Applies Fourier analysis to transform protein sequences into a frequency domain, allowing for the detection of periodic patterns indicative of repeats.
* **Complexity Measurement Approaches (Shannon Entropy)**: Quantifies the complexity of protein regions using metrics like Shannon entropy to identify Low Complexity Regions (LCRs), which often correspond to repeat sequences.
* **General Repeat Detection**: A core script that likely integrates or orchestrates different repeat detection methods, including options for both exact and approximate matching using substitution matrices like BLOSUM62.

---

### Files Included

* **`slides.pdf`**: The main presentation slides, providing an overview of amino acid repeats, their classification, detection strategies, and functional implications.
* **`fourier transform.ipynb`**: Python script for detecting repeats using Fourier transform.
* **`self alignment.ipynb`**: Python script for repeat detection via self-comparison alignment.
* **`shannon entropy.ipynb`**: Python script for identifying low complexity regions using Shannon entropy.
* **`repeat detector.py`**: A general-purpose Python script for finding amino acid repeats, potentially incorporating exact and approximate matching algorithms.

---

### Usage

To use the Python scripts, you will need a Python environment with common scientific libraries (e.g., NumPy, Matplotlib) and Biopython for sequence alignment matrices. Each Python file typically contains example usage within the script, demonstrating how to apply the detection methods to a `test_sequence`.

---

### Author

Ali Ahmadi Esfidi