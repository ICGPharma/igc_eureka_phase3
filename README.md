# IGC Eureka Phase 3 – Speech-Based ADRD Detection

This repository contains all scripts, and data workflows used in Phase 3 of the IGC Pharma submission to the PREPARE Challenge. Our project focuses on the detection of Alzheimer's Disease and Related Dementias (ADRD) using speech-based models, with a particular emphasis on model generalization, interpretability, and multilingual inclusion.

## Repository Structure

```
igc_phase3/
├── data/                  
│   ├── raw/               # Original downloaded datasets
│   ├── interim/           # Intermediate processing results
│   └── processed/         # Cleaned and final datasets used for training
│
├── notebooks/             # Jupyter notebooks for exploration and evaluation
│
├── src
│   └── data/             
│       ├── 01_raw_data.py                  # Script to load and inspect raw audio data
│       ├── 02_split_data.py                # Train/test split strategy, including benchmark splits
│       ├── 03_extract_metadata.py          # Extraction of speaker age, gender, and education.
│       ├── 04_identify_tasks.py            # Identify and annotate tasks per recording
│       ├── 05_extract_acustic_features.py  # Low-level audio feature extraction
│       ├── 06_STT_nonenglish_audios.py     # Speech-to-text and translation for non-English recordings
│       └── 07_TTS_nonenglish_audios.py     # Speech synthesis to English
│        
│   └── model/
├── LICENSE                 
└── README.md     
```
## Accessing the Data

Access to DementiaBank data requires authentication via a session cookie. To download data using 01_raw_data.py, follow these steps:

1. Create a free account at TalkBank.
2. Request access to the specific DementiaBank studies you intend to use.
3. Log in at https://talkbank.org/dementia/.
4. Open your browser’s developer tools (e.g., right-click → “Inspect” or press F12), then:
    * Go to the Network tab
    * Refresh the page
    * Click on the isLoggedIn request
    * Under the Cookies section, locate and copy the value of the talkbank cookie.

5. Run the script with the cookie:
```
python src/data/01_raw_data.py --cookie "your_cookie_value_here"
```

### Note

All data used in this project was obtained from the DementiaBank collection within the TalkBank repository. The data partitioning described here is based on the dataset as it existed on *May 9, 2025*. Any additions or changes to the dataset made after this date are not reflected in the current splits used for the challenge.

If needed, the partitioning process can be reproduced or updated using the script located at:
src/data/02_split_data.py
